import io
import requests
import pandas as pd

from unittest import TestCase


class ServiceTest(TestCase):
    server_url: str = "http://127.0.0.1:8000"
    data_url: str = "https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv"

    def testPredictItem(self):
        path = "/predict_item"
        df = pd.read_csv(self.data_url)
        item = df.sample(1).to_dict(orient="records")
        resp = requests.post(self.server_url + path, json=item[0])
        assert resp.status_code == 200

    def testPredictItems(self):
        path = "/predict_items"
        df = pd.read_csv(self.data_url)
        df.sample(10).drop(columns="selling_price").to_csv(
            buffer := io.StringIO(), index=False
        )
        resp = requests.post(self.server_url + path, files={"csv": buffer.getvalue()})
        assert resp.status_code == 200
