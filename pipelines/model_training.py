import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
from pathlib import Path
import yaml
import logging
import os
from dotenv import load_dotenv
import tempfile
import shutil
import pickle
import json

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTraining:
    
    def __init__(self, config_path: str, feature_engineer=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        self.training_config = self.config['training']
        self.model_params = self.config['model_params']
        self.mlflow_config = self.config['mlflow']
        self.feature_engineer = feature_engineer
        
        experiment_name = self.mlflow_config['experiment_name']
        mlflow.set_experiment(experiment_name=experiment_name)

        self.artifact_uri = (
            f"wasbs://{os.getenv('AZURE_STORAGE_CONTAINER_NAME')}"
            f"@{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/mlflow/{experiment_name}"
        )

        if Path("/run/secrets/azure_storage_container_name").exists():
            azure_container = Path("/run/secrets/azure_storage_container_name").read_text().strip()
            azure_account = Path("/run/secrets/azure_storage_account_name").read_text().strip()
            self.artifact_uri = (
                f"wasbs://{azure_container}@{azure_account}.blob.core.windows.net/mlflow/{experiment_name}"
            )

        mlflow_client = mlflow.tracking.MlflowClient()
        exp = mlflow_client.get_experiment_by_name(experiment_name)
        if exp is None:
            self.experiment_id = mlflow_client.create_experiment(
                name=experiment_name,
                artifact_location=self.artifact_uri
            )
        else:
            self.experiment_id = exp.experiment_id
        
        logger.info(f"MLflow Experiment initialized: {experiment_name}")
        logger.info(f"Artifact URI: {self.artifact_uri}")


    def get_model(self, model_type: str):
        logger.info(f"Creating {model_type} model")

        if model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params['random_forest'])
        elif model_type == 'xgboost':
            return XGBClassifier(**self.model_params['xgboost'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        best_model = None
        self.best_score = -np.inf
        best_model_type = None

        for model_type in self.training_config['model_types']:
            logger.info(f"Training model: {model_type}")

            with mlflow.start_run(
                run_name=f"{self.mlflow_config['run_name']}_{model_type}",
                experiment_id=self.experiment_id,
                nested=True
            ) as run:

                run_id = run.info.run_id
                mlflow.log_param("model_type", model_type)
                mlflow.log_params(self.training_config)
                mlflow.log_params(self.model_params[model_type])

                model = self.get_model(model_type)

                unique_labels = y_train.unique()
                if not set(unique_labels).issubset({0, 1}):
                    logger.warning(f"Expected y_train to be encoded as 0/1, got: {unique_labels}")
                    y_clean = y_train.astype(str).str.strip().str.capitalize()
                    y_encoded = y_clean.map({'No': 0, 'Yes': 1})
                    if y_encoded.isna().any():
                        raise ValueError(f"Invalid labels: {y_clean[y_encoded.isna()].unique()}")
                    y_encoded = y_encoded.astype(int)
                else:
                    y_encoded = y_train.astype(int)

                X_train_transformed = self.feature_engineer.transform(X_train.copy())
                model.fit(X_train_transformed, y_encoded)

                cv_scores = cross_val_score(
                    model,
                    X_train_transformed,
                    y_encoded,
                    cv=self.training_config['cv_folds'],
                    scoring=self.training_config['scoring'],
                    n_jobs=self.training_config['n_jobs']
                )

                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                mlflow.log_metric("cv_mean_score", mean_score)
                mlflow.log_metric("cv_std_score", std_score)

                logger.info(f"{model_type} {self.training_config['scoring']}: "
                            f"{mean_score:.4f} (+/- {std_score:.4f})")

                model_pipeline = Pipeline([
                    ('preprocessor', self.feature_engineer),
                    (model_type, model)
                ])
                X_example = X_train.iloc[:5].copy()
                y_pred_example = model_pipeline.predict(X_example)
                signature = infer_signature(X_example, y_pred_example)

                mlflow.sklearn.log_model(
                    sk_model=model_pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_example
                )

                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp_config:
                    json.dump(self.config, tmp_config, indent=2)
                    tmp_config_path = tmp_config.name
                mlflow.log_artifact(tmp_config_path, artifact_path="config")
                os.remove(tmp_config_path)

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    best_model = model_pipeline
                    best_model_type = model_type
                    best_run_id = run_id

                logger.info(f"Current best model: {best_model_type} ({self.best_score:.4f})")

        logger.info(f"Best model selected: {best_model_type} with score {self.best_score:.4f}")
        return best_model, best_model_type, best_run_id

    def save_all_artifacts_to_mlflow(self, X_train: pd.DataFrame, y_train: pd.Series):
        best_model, best_model_type, best_run_id = self.train(X_train, y_train)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=f"best_model_{best_model_type}", experiment_id=self.experiment_id,nested=True) as run:
            run_id = run.info.run_id
            X_example = X_train.iloc[:5].copy()

            y_pred_example = best_model.predict(X_example)

            signature = infer_signature(X_example, y_pred_example)


            mlflow.log_param("best_model_type", best_model_type)
            mlflow.log_metric("best_cv_score", self.best_score)
            mlflow.log_param("timestamp", timestamp)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name=f"{best_model_type}_final_model",
                signature=signature,
                input_example=X_example
            )

            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pkl") as tmp_file:
                pickle.dump(best_model, tmp_file)
                tmp_path = tmp_file.name
            mlflow.log_artifact(tmp_path, artifact_path="pickled_model")
            os.remove(tmp_path)

            metrics_path = os.path.join(tempfile.gettempdir(), "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({
                    "best_model_type": best_model_type,
                    "best_cv_score": self.best_score,
                    "timestamp": timestamp,
                    "best_run_id": best_run_id
                }, f, indent=2)
            mlflow.log_artifact(metrics_path, artifact_path="metadata")

            logger.info(f"Best model logged to MLflow with run_id={run_id}")

        temp_dir = tempfile.mkdtemp(prefix="mlflow_download_")
        downloaded_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/",
            dst_path=temp_dir
        )

        logger.info(f"📥 Artifacts downloaded locally to: {downloaded_dir}")

        return {
            "best_model": best_model,
            "best_model_type": best_model_type,
            "best_score": self.best_score,
            "best_run_id": best_run_id,
            "final_run_id": run_id,
            "local_artifact_dir": downloaded_dir
        }
