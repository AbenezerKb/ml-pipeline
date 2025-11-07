import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import logging
import os
from datetime import datetime, timezone
from pipelines.pre_processing import DataPreparation,preprocess_data, get_preprocessor



SAMPLE_CONFIG = """
data:
  test_size: 0.2
  random_state: 42
preprocessing:
  handle_missing: mean
  handle_outliers: true
features:
  numerical: ['num1', 'num2']
  target: Chrun
"""

SAMPLE_DF = pd.DataFrame({
    'num1': [1, 2, np.nan, 4, 100],
    'num2': [10, 20, 30, 40, 50],
    'cat': ['A', 'B', np.nan, 'A', 'B'],
    'target': [0, 1, 0, 1, 0]
})

@pytest.fixture
def mock_config_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_CONFIG)
    return str(config_path)

@pytest.fixture
def data_prep(mock_config_file):
    return DataPreparation(mock_config_file)

@patch('pandas.read_csv')
@patch('pathlib.Path.exists', return_value=False)
@patch('pathlib.Path.read_text', return_value='data/sample.csv')
def test_load_data(mock_read_text, mock_exists, mock_read_csv, data_prep):
    mock_read_csv.return_value = SAMPLE_DF
    df = data_prep.load_data()
    assert len(df) == 5
    assert list(df.columns) == ['num1', 'num2', 'cat', 'target']
    mock_read_csv.assert_called_with("data/sample.csv")

def test_handle_missing_values_mean(data_prep):
    df = SAMPLE_DF.copy()
    imputed_df = data_prep.handle_missing_values(df)
    assert imputed_df['num1'].isnull().sum() == 0
    assert imputed_df['num1'][2] == pytest.approx((1 + 2 + 4 + 100) / 4)  # Mean
    assert imputed_df['cat'][2] == 'A'

def test_handle_missing_values_median(data_prep):
    data_prep.preprocessing_config['handle_missing'] = 'median'
    df = SAMPLE_DF.copy()
    imputed_df = data_prep.handle_missing_values(df)
    assert imputed_df['num1'][2] == np.median([1, 2, 4, 100])  # 3.0

def test_handle_outliers(data_prep):
    df = SAMPLE_DF.copy()
    df = data_prep.handle_missing_values(df)
    outlier_df = data_prep.handle_outliers(df, ['num1', 'num2'])

    assert outlier_df['num1'][4] == pytest.approx(63.875)


@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pipelines.pre_processing.datetime')  # patch the module's datetime object
def test_save_processed_data(mock_datetime_module, mock_to_csv, mock_makedirs, data_prep):
    fake_dt = datetime(2023, 1, 1, 0, 0, 0)
    mock_datetime_module.now.return_value = fake_dt
    mock_datetime_module.utcnow.return_value = fake_dt

   
    mock_datetime_module.strftime = datetime.strftime

    train_df = SAMPLE_DF.iloc[:3]
    test_df = SAMPLE_DF.iloc[3:]
    data_prep.save_processed_data(train_df, test_df)

    mock_makedirs.assert_called_with(
        os.path.join("data", "processed", "20230101_000000"),
        exist_ok=True
    )
    assert mock_to_csv.call_count == 2

@patch.object(DataPreparation, 'save_processed_data')
@patch.object(DataPreparation, 'split_data', return_value=(SAMPLE_DF.iloc[:3], SAMPLE_DF.iloc[3:]))
@patch.object(DataPreparation, 'handle_outliers', return_value=SAMPLE_DF)
@patch.object(DataPreparation, 'handle_missing_values', return_value=SAMPLE_DF)
@patch.object(DataPreparation, 'load_data', return_value=SAMPLE_DF)
def test_run(mock_load_data, mock_handle_missing_values, mock_handle_outliers, mock_split_data, mock_save_processed_data, data_prep):
    train_df, test_df = data_prep.run()
    assert len(train_df) == 3
    assert len(test_df) == 2
    mock_load_data.assert_called_once()
    mock_handle_missing_values.assert_called_once()
    mock_handle_outliers.assert_called_once_with(SAMPLE_DF, ['num1', 'num2'])
    mock_split_data.assert_called_once()
    mock_save_processed_data.assert_called_once()


### New test for new preprocessing

# Sample data similar to X_train
SAMPLE_DF_PREPROCESS = pd.DataFrame({
    "Age": [25, 40, np.nan],
    "Income": [40000, 80000, 60000],
    "Gender": ["Male", "Female", "Female"],
    "City": ["London", "Paris", None]
})

@pytest.fixture
def sample_data_preprocess():
    return SAMPLE_DF_PREPROCESS.copy()

def test_get_preprocessor_returns_column_transformer(sample_data_preprocess):
    preprocessor = get_preprocessor()
    assert hasattr(preprocessor, "fit_transform")  # Vérifie que c'est bien un transformer

    X_transformed = preprocessor.fit_transform(sample_data_preprocess)
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == sample_data_preprocess.shape[0]

def test_preprocess_data_function(sample_data_preprocess):
    X_processed = preprocess_data(sample_data_preprocess)
    assert isinstance(X_processed, pd.DataFrame)
    assert X_processed.shape[0] == sample_data_preprocess.shape[0]

    # Vérifie qu'il n'y a plus de NaN sur les colonnes numériques
    numeric_cols = sample_data_preprocess.select_dtypes(include=['float64', 'int64']).columns
    assert not X_processed[numeric_cols].isna().any().any()

def test_ordinal_encoding_behavior(sample_data_preprocess):
    X_processed = preprocess_data(sample_data_preprocess)
    # Les colonnes catégorielles doivent être encodées en nombres
    assert np.issubdtype(X_processed["Gender"].dtype, np.number)
    # Vérifie que le None de "City" a été transformé en -1 par l'OrdinalEncoder
    assert -1 in X_processed["City"].values
