import pytest
import pandas as pd
from pipelines.feature_selection import feature_selection


def test_feature_selection_removes_columns():
    """Test that the columns are properly removed"""
   
    df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'TruckOwner': [0, 1, 0],
        'MonthlyRevenue': [50.5, 60.2, 45.8],
        'Churn': [0, 1, 0]
    })
    
  
    result = feature_selection(df)
    
    
    assert 'CustomerID' not in result.columns
    assert 'TruckOwner' not in result.columns
    
    assert 'MonthlyRevenue' in result.columns
    assert 'Churn' in result.columns


def test_feature_selection_copy_parameter():
    """Test that the copy parameter works correctly"""
    df_original = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'MonthlyRevenue': [50.5, 60.2, 45.8]
    })
    
   
    df_copy = df_original.copy()
    result = feature_selection(df_copy, copy=True)
    assert id(df_copy) != id(result), "A new DataFrame should be created"
    assert 'CustomerID' in df_copy.columns, "The original DataFrame should not be modified when copy=True"
    assert 'CustomerID' not in result.columns, "CustomerID should be removed from the result"
    
    df_no_copy = df_original.copy()
    result = feature_selection(df_no_copy, copy=False)
    
    assert 'CustomerID' not in result.columns, "CustomerID should be removed from the result"


def test_feature_selection_no_columns_to_drop():
    """Test when no columns should be removed"""
    df = pd.DataFrame({
        'MonthlyRevenue': [50.5, 60.2, 45.8],
        'Churn': [0, 1, 0]
    })
    
    result = feature_selection(df)
    
   
    assert len(result.columns) == len(df.columns)


def test_feature_selection_empty_dataframe():
    """Test with an empty DataFrame"""
    df = pd.DataFrame()
    
    result = feature_selection(df)
    
    assert result.empty
    assert len(result.columns) == 0