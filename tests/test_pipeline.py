import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pipelines.pipeline import pipeline


def test_pipeline_calls_feature_selection_and_engineering():
    """"Test that the pipeline correctly calls both preprocessing functions"""
    df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'CustomerCareCalls': [5, 10, 2],
        'RetentionCalls': [2, 4, 1],
        'MonthlyRevenue': [50.5, 60.2, 45.8]
    })
    
    # None by default
    result = pipeline(df)
    
   
    assert 'CustomerID' not in result.columns
    
    #check that feature engineering was applied
    assert 'TotalSupportCalls' in result.columns


def test_pipeline_copy_parameter():
    """Test that the copy parameter works"""
    df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'MonthlyRevenue': [50.5, 60.2, 45.8]
    })
    
    # Test with copy=True
    result = pipeline(df, copy=True)
    assert id(df) != id(result)
    assert 'CustomerID' in df.columns, "Le DataFrame original ne devrait pas être modifié"


def test_pipeline_with_mock_preprocessor():
    """Test du pipeline avec un preprocessor mocké"""
    df = pd.DataFrame({
        'MonthlyRevenue': [50.5, 60.2, 45.8],
        'CustomerCareCalls': [5, 10, 2],
        'RetentionCalls': [2, 4, 1]
    })
    
    
    mock_preprocessor = Mock()
    mock_preprocessor.fit_transform.return_value = df.copy()
    mock_preprocessor.transform.return_value = df.copy()
    
   
    with patch('pipelines.pipeline.preprocessor', mock_preprocessor):
       
        result = pipeline(df, train=True)
        mock_preprocessor.fit_transform.assert_called_once()
        
       
        mock_preprocessor.reset_mock()
        
        result = pipeline(df, train=False)
        mock_preprocessor.transform.assert_called_once()


def test_pipeline_without_preprocessor():
    """Test that the pipeline works even without a preprocessor"""
    df = pd.DataFrame({
        'CustomerCareCalls': [5, 10, 2],
        'RetentionCalls': [2, 4, 1],
        'MonthlyRevenue': [50.5, 60.2, 45.8]
    })
    
    
    result = pipeline(df)
    
    # Try to check that feature selection and engineering were applied
    assert 'TotalSupportCalls' in result.columns
    assert isinstance(result, pd.DataFrame)


def test_pipeline_empty_dataframe():
    """Test with an empty DataFrame"""
    df = pd.DataFrame()
    
    result = pipeline(df)
    
    assert result.empty