import pickle

import numpy as np
from fastapi import FastAPI
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from starlette.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
@app.get("/")
async def hello():
    return {
        "hello!"
        "/train will train a logistic regression"
        "/predict will predict using trained model"
    }
@app.get("/train")
async  def train():
    # LOAD DATASET
    data = pd.read_csv("data.csv")

    data.head()

    data.info()

    data.drop(columns=['No'], inplace=True)

    data.describe()

    # ATTRIBUTES ONLY
    x = data.drop(columns=['Y house price of unit area'])
    # PREDICTION TARGET
    y = data['Y house price of unit area']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # SET LINEAR REGRESSION LEARNING MODEL
    model = LinearRegression()

    # TRAIN MODEL
    model.fit(x_train, y_train)
    # TEST MODEL
    model.predict(x_test)
    # DISPLAY R^2 SCORE
    print('R2 score is ', r2_score(y_test, model.predict(x_test)))
    # INITIALIZE BEST-FIT LINE AND ACTUAL VALUES GRAPH
    plt.scatter(y_test, model.predict(x_test), color='blue', label='Actual')

    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Best Fit Line')

    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.title('Actual vs Predicted Y')
    plt.legend()
    # SAVE PLOT
    plot_file = "plot.png"
    plt.savefig(plot_file)
    # DISPLAY PLOT
    plt.show()
    # RETURN PLOT AS ENDPOINT RESPONSE
    return FileResponse(path="plot.png", media_type="image/png")



