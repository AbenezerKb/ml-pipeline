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
    
    