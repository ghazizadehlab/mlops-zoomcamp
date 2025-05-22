import os
import pickle
import click
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    # X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # rf = RandomForestRegressor(max_depth=10, random_state=0)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_val)

    # rmse = mean_squared_error(y_val, y_pred, squared=False)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    # Enable autologging
    mlflow.sklearn.autolog()
    # mlflow.log_param("data_path", "./output")
    
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        
        # Log parameters
        mlflow.log_param("max_depth", 10)
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # Calculate RMSE manually
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"RMSE: {rmse}")



if __name__ == '__main__':
    run_train()
