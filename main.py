import pickle
import uvicorn

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from src.features_processor import process_features
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
def predict_item(item: Item, feature_dict: Dict[str, Any] = Depends(get_feature_dict),
                 column_transformer: ColumnTransformer = Depends(get_column_transformer),
                 model: LinearRegression = Depends(get_model)) -> float:
    processed_features = process_features([item], feature_dict)
    X = column_transformer.transform(processed_features)
    y = model.predict(X)
    return 10 ** y


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    pass
    # return ...


if __name__ == "__main__":
    uvicorn.run("main:app")
