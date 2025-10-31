import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import yaml
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.preprocessing_config = self.config['preprocessing']
        self.raw_data = self.data_config['raw_data']
    
    def load_data(self) -> pd.DataFrame:    
        logger.info("Loading data")
        df = pd.read_csv(self.data_config['raw_data'])
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.preprocessing_config['handle_missing']
        logger.info(f"Handling missing values using {method} method")
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    if method == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        if not self.preprocessing_config['handle_outliers']:
            return df
        
        logger.info("Handling outliers using IQR method")
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        
        test_size = self.data_config['test_size']
        random_state = self.data_config['random_state']
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self.config['features']['target']] if self.config['features']['target'] in df.columns else None
        )
        
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):       
        logger.info("Saving processed data to MongoDB")
        
        timestamp = datetime.utcnow()
        processing_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        train_records = train_df.to_dict('records')
        for record in train_records:
            record['split'] = 'train'
            record['processing_id'] = processing_id
            record['processed_at'] = timestamp
        
        test_records = test_df.to_dict('records')
        for record in test_records:
            record['split'] = 'test'
            record['processing_id'] = processing_id
            record['processed_at'] = timestamp
        
        processed_collection = self.db['processed_data']
        
        if train_records:
            train_result = processed_collection.insert_many(train_records)
            logger.info(f"Saved {len(train_result.inserted_ids)} train records")
        
        if test_records:
            test_result = processed_collection.insert_many(test_records)
            logger.info(f"Saved {len(test_result.inserted_ids)} test records")
        
        metadata_collection = self.db['processing_metadata']
        metadata = {
            'processing_id': processing_id,
            'timestamp': timestamp,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'total_size': len(train_df) + len(test_df),
            'train_columns': list(train_df.columns),
            'test_columns': list(test_df.columns),
            'config': self.config,
            'status': 'completed'
        }
        metadata_collection.insert_one(metadata)
        
        logger.info(f"Processing metadata saved with ID: {processing_id}")
        
        return {
            'processing_id': processing_id,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'collection': 'processed_data'
        }
    
    def run(self):
        df = self.load_data()
        
        df = self.handle_missing_values(df)
        
        numerical_cols = self.config['features']['numerical']
        df = self.handle_outliers(df, numerical_cols)
        
        train_df, test_df = self.split_data(df)
        
        self.save_processed_data(train_df, test_df)
        
        logger.info("Data preparation completed successfully")
        return train_df, test_df