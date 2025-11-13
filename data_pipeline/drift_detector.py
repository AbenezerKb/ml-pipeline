import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DriftDetector:    
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.reference_data_path = config.get('reference_data_path')
        self.metrics = config.get('metrics', ['psi', 'ks', 'wasserstein'])
        self.thresholds = config.get('thresholds', {})
        self.categorical_columns = config.get('categorical_columns', [])
        self.numerical_columns = config.get('numerical_columns', [])
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return numerical, categorical
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
       
        try:
            
            if reference.dtype == 'object' or current.dtype == 'object':
                reference_counts = reference.value_counts(normalize=True)
                current_counts = current.value_counts(normalize=True)
                
                
                all_categories = reference_counts.index.union(current_counts.index)
                reference_dist = reference_counts.reindex(all_categories, fill_value=0.001)
                current_dist = current_counts.reindex(all_categories, fill_value=0.001)
            else:
                
                min_val = min(reference.min(), current.min())
                max_val = max(reference.max(), current.max())
                bin_edges = np.linspace(min_val, max_val, bins + 1)
                
                reference_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
                current_counts, _ = np.histogram(current.dropna(), bins=bin_edges)
                
                reference_dist = (reference_counts + 0.001) / (reference_counts.sum() + 0.001 * bins)
                current_dist = (current_counts + 0.001) / (current_counts.sum() + 0.001 * bins)
                        
            psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return np.nan
    
    def _calculate_ks_statistic(self, reference: pd.Series, current: pd.Series) -> float:
       
        try:

            if reference.dtype == 'object' or current.dtype == 'object':
                return np.nan
            
            reference_clean = reference.dropna()
            current_clean = current.dropna()
            
            if len(reference_clean) == 0 or len(current_clean) == 0:
                return np.nan
            
            statistic, _ = stats.ks_2samp(reference_clean, current_clean)
            return float(statistic)
            
        except Exception as e:
            logger.error(f"Error calculating KS statistic: {e}")
            return np.nan
    
    def _calculate_wasserstein_distance(self, reference: pd.Series, current: pd.Series) -> float:
       
        try:

            if reference.dtype == 'object' or current.dtype == 'object':
                return np.nan
            
            reference_clean = reference.dropna()
            current_clean = current.dropna()
            
            if len(reference_clean) == 0 or len(current_clean) == 0:
                return np.nan
            
            distance = stats.wasserstein_distance(reference_clean, current_clean)
            
            
            data_range = max(reference_clean.max(), current_clean.max()) - min(reference_clean.min(), current_clean.min())
            if data_range > 0:
                distance = distance / data_range
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return np.nan
    
    def _calculate_chi_square(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:

        try:

            reference_counts = reference.value_counts()
            current_counts = current.value_counts()
                        
            all_categories = reference_counts.index.union(current_counts.index)
            reference_aligned = reference_counts.reindex(all_categories, fill_value=0)
            current_aligned = current_counts.reindex(all_categories, fill_value=0)
            
            contingency_table = pd.DataFrame({
                'reference': reference_aligned,
                'current': current_aligned
            })
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table.T)
            
            return float(chi2), float(p_value)
            
        except Exception as e:
            logger.error(f"Error calculating chi-square: {e}")
            return np.nan, np.nan
    
    async def detect_drift(self, current_data_path: pd.DataFrame, reference_data_path: pd.DataFrame) -> Dict[str, Any]:
       
        if not self.enabled:
            logger.info("Drift detection is disabled")
            return False
        
        
        try:

            if 'Churn' in reference_data_path.columns:
                reference_data_path =reference_data_path.drop(columns=['Churn'])
            if 'Churn' in current_data_path.columns:    
                current_data_path =current_data_path.drop(columns=['Churn'])
            common_columns = list(set(reference_data_path.columns) & set(current_data_path.columns))


            reference_data_path = reference_data_path[common_columns]
            current_data_path = current_data_path[common_columns]          
            
            drifted_columns = []
            drift_report = Report(metrics=[DataDriftPreset()])
            result = drift_report.run(reference_data=reference_data_path, current_data=current_data_path)
            important_columns = ['CurrentEquipmentDays','MonthsInService', 'RetentionCalls', 'HandSetWebCapable', 'RespondsToMailOffers', 'MonthlyMinutes',
            'CreditRating', 'PercChangeMinutes', 'UniqueSubs', 'TotalRecurringCharge', 'BuysViaMailOrder', 'Homeownership']
            for metric in result.dict()['metrics'][1:]:             
                if metric['metric_name'].startswith('ValueDrift'):
                    column_name = metric['config']['column']
                    drift_value = metric['value']
                    threshold = metric['config']['threshold']
                  
                    if drift_value > threshold and column_name in important_columns:
                        drifted_columns.append({
                            'column': column_name,
                            'drift_score': drift_value,
                            'threshold': threshold,
                            'method': metric['config']['method']
                        })

            drift_detected = len(drifted_columns) > 0
            
            result.save_html("simulated_data_drift_report.html")
            
            
            logger.info(f"Drift detection complete")
           
            
        except Exception as e:
            logger.error(f"Error during drift detection: {e}", exc_info=True)
            result['error'] = str(e)
            result['drift_detected'] = False
        
        return drift_detected