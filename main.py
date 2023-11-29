import io
import pickle
import uvicorn
import pandas as pd

from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from src.features_processor import process_features, preprocess_collection
from src.identity_transformer import IdentityTransformer

MODEL_PATH = "assets/model.pickle"
with open(MODEL_PATH, "rb") as f:
    model_dict = pickle.load(f)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    # selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def get_feature_dict():
    return {"medians": model_dict["medians"], "year": model_dict["year"]}


def get_column_transformer():
    return model_dict["column_transformer"]


def get_model():
    return model_dict["model"]


@app.post("/predict_item")
def predict_item(
    item: Item,
    feature_dict: Dict[str, Any] = Depends(get_feature_dict),
    column_transformer: ColumnTransformer = Depends(get_column_transformer),
    model: LinearRegression = Depends(get_model),
) -> float:
    df = preprocess_collection([item])
    processed_features = process_features(df, feature_dict)
    X = column_transformer.transform(processed_features)
    y = 10 ** model.predict(X)
    return y


@app.post("/predict_items")
def predict_items(
    csv: UploadFile,
    feature_dict: Dict[str, Any] = Depends(get_feature_dict),
    column_transformer: ColumnTransformer = Depends(get_column_transformer),
    model: LinearRegression = Depends(get_model),
) -> List[float]:
    df = pd.read_csv(csv.file)
    processed_features = process_features(df, feature_dict)
    X = column_transformer.transform(processed_features)
    y = 10 ** model.predict(X)
    df["selling_price"] = y
    # df.to_csv(buffer := io.StringIO(), index=False)
    return StreamingResponse(
        io.StringIO(df.to_csv(index=False)),
        media_type="text/csv",
    )


if __name__ == "__main__":
    uvicorn.run("main:app")
