from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import src.util as util
import src.data_pipeline as data_pipeline
import src.preprocessing as preprocessing

config_data = util.load_config()
ohe_continent = util.pickle_load(config_data["ohe_continent_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    hdi : float
    continent : object
    EFConsPerCap : float


app = FastAPI()

# general endpoint (testing endpoint)
@app.get("/")
def home():
    return "Hello, FastAPI up!"

# health check
@app.get("/health/")
def home():
    return "200"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # Convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][0]],
            data[config_data["predictors"][1:]]
        ],
        axis = 1
    )

    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    #preprocessing data in serving
    # Encoding continent
    data = preprocessing.ohe_transform(data, "continent", ohe_continent)

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
