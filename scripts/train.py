import sys
from pathlib import Path
import logging
import argparse
import joblib
import os
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature

from pipelines.pre_processing import DataPreparation
from pipelines.feature_engineering import FeatureEngineering
from pipelines.model_training import ModelTraining
from pipelines.model_evaluation import ModelEvaluation
from dotenv import load_dotenv
import mlflow
import yaml
import pandas as pd
from datetime import datetime
import tempfile

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train(config_path: str):
    logger.info("Starting ML Training Pipeline")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        os.environ['AZURE_STORAGE_ACCOUNT_NAME'] = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        os.environ['AZURE_STORAGE_ACCOUNT_KEY'] = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        if Path("/run/secrets/azure_connection_string").exists():
            azure_connection_string = Path("/run/secrets/azure_connection_string").read_text().strip() 
            azure_container = Path("/run/secrets/azure_container_name").read_text().strip() if Path("/run/secrets/azure_container_name").exists() else azure_container
        tracking_uri = os.getenv('REGISTRY_URI')
        mlflow.set_tracking_uri(tracking_uri)

    try:
        logger.info("\nData Preparation")
        
        data_prep = DataPreparation(config_path)
        train_df, test_df = data_prep.run()

        logger.info("\nFeature Engineering")
        
        feature_eng = FeatureEngineering(config_path)
        target_col = data_prep.config['features']['target']

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        def encode_target(y):
            y_clean = y.astype(str).str.strip().str.capitalize()
            y_encoded = y_clean.map({'No': 0, 'Yes': 1})
            if y_encoded.isna().any():
                raise ValueError(f"Invalid labels: {y_clean[y_encoded.isna()].unique()}")
            return y_encoded.astype(int)
        
        y_train_encoded = encode_target(y_train)
        y_test_encoded = encode_target(y_test)

        logger.info(f"Original X_train columns: {list(X_train.columns)}")
        logger.info(f"Original X_test columns: {list(X_test.columns)}")
        
        
        feature_eng.fit(X_train)
        
        
        X_train_transformed = feature_eng.transform(X_train)
        X_test_transformed = feature_eng.transform(X_test)
        
        logger.info(f"Transformed train shape: {X_train_transformed.shape}")
        logger.info(f"Transformed train columns: {list(X_train_transformed.columns)}")
        logger.info(f"Transformed test shape: {X_test_transformed.shape}")
        logger.info(f"Transformed test columns: {list(X_test_transformed.columns)}")
        
        
        if list(X_train_transformed.columns) != list(X_test_transformed.columns):
            logger.error("COLUMN MISMATCH DETECTED!")
            logger.error(f"Train columns: {set(X_train_transformed.columns)}")
            logger.error(f"Test columns: {set(X_test_transformed.columns)}")
            logger.error(f"Only in train: {set(X_train_transformed.columns) - set(X_test_transformed.columns)}")
            logger.error(f"Only in test: {set(X_test_transformed.columns) - set(X_train_transformed.columns)}")
            raise ValueError("Column mismatch between train and test sets")
        
        
        artifact_dir = Path(tempfile.mkdtemp(prefix="mlflow-artifacts-"))
        fe_path = artifact_dir / "feature_engineer.joblib"
        joblib.dump(feature_eng, fe_path)
        feature_eng.save(Path("artifact"))
        
        
        
        model_trainer = ModelTraining(config_path, feature_engineer=feature_eng)
                
        
        result = model_trainer.save_all_artifacts_to_mlflow(X_train, y_train_encoded)
        model, model_type = result["best_model"], result["best_model_type"]

        logger.info(f"\nModel Evaluation for {model_type}")
        
        
        evaluator = ModelEvaluation(config_path)
        logger.info("Evaluating model on TRANSFORMED test data")
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        result = X_test.copy()
        
        meets_thresholds = evaluator.check_thresholds(metrics)

        
        result["prediction"] = predictions
        result["probability"] = probabilities
        now = datetime.now()
        result.to_csv(f"data/predictions_{now}.csv", index=False)
        mlflow.log_artifact(f"data/predictions_{now}.csv", artifact_path="mlflow-artifacts")
        logger.info("Done! Predictions saved.")        

        logger.info("Cleaning Up Local")
        os.remove(f"data/predictions_{now}.csv")
        
        if meets_thresholds:
            logger.info("\nModel meets all performance thresholds!")
        else:
            logger.warning("\nModel does not meet some performance thresholds")
                     
        logger.info("Training Pipeline Completed Successfully!")
       
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise