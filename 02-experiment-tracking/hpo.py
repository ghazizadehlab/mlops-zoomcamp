import os
import pickle
import click
import mlflow
import numpy as np
import yaml
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)

def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)
            
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            # Log metric
            mlflow.log_metric("validation_rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("random-forest-hyperopt")

    with mlflow.start_run():  # outer run
        rstate = np.random.default_rng(42)
        best =fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
            rstate=rstate
        )
        mlflow.log_params(best)  # log best trial globally
        
        print(best)
        search_space_yaml_friendly = {
            'max_depth': 'int ∈ [1, 20]',
            'n_estimators': 'int ∈ [10, 50]',
            'min_samples_split': 'int ∈ [2, 10]',
            'min_samples_leaf': 'int ∈ [1, 4]',
            'random_state': 42
        }

        with open("search_space.yaml", "w") as f:
            yaml.dump(search_space_yaml_friendly, f)

        mlflow.log_artifact("search_space.yaml")

if __name__ == '__main__':
    run_optimization()
