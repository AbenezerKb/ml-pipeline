import pytest
import pandas as pd
import numpy as np
import json
import pytest
import tempfile
from pathlib import Path
from pipelines.feature_engineering import FeatureEngineering
from pipelines.feature_engineering import feature_engineering
@pytest.fixture
def sample_config():

    return {
        "features": {
            "numerical": ["Age", "Income"],
            "categorical": ["Gender", "City"],
            "selected_features": ["Age", "Income", "Gender", "City"]
        },
        "encoding": {
            "Gender": {
                "strategy": "ordinal",
                "categories": ["Male", "Female"]
            },
            "City": {
                "strategy": "onehot",
                "categories": ["London", "Paris"]
            }
        },
        "preprocessing": {
            "scaler": "standard"
        }
    }


@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "Age": [25, 40, 35],
        "Income": [40000, 80000, 60000],
        "Gender": ["Male", "Female", "Female"],
        "City": ["London", "Paris", "London"],
        "InboundCalls": [5, 2, 4],
        "OutboundCalls": [3, 6, 1]
    })


def test_fit_and_transform(sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)

    transformed = fe.transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)

    assert "TotalCalls" in fe.numerical

    assert abs(transformed["Age"].mean()) < 1e-6

    assert any(c.startswith("City_") for c in transformed.columns)

    names = fe.get_feature_names_out()
    assert all(name in transformed.columns for name in names)


def test_invalid_config_raises(sample_data):

    bad_config = {
        "features": {"numerical": ["Age"], "categorical": ["Color"]},
        "encoding": {"Color": {"strategy": "ordinal"}}
    }
    with pytest.raises(ValueError, match="Missing 'categories'"):
        FeatureEngineering(bad_config).fit(sample_data)


def test_scaling_modes(sample_config, sample_data):

    for scaler in ["standard", "minmax", "robust"]:
        config = sample_config.copy()
        config["preprocessing"]["scaler"] = scaler
        fe = FeatureEngineering(config)
        fe.fit(sample_data)
        transformed = fe.transform(sample_data)
        assert not transformed.isna().any().any(), f"{scaler} introduced NaNs"


def test_ordinal_and_onehot_encoding_behavior(sample_config, sample_data):

    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    transformed = fe.transform(sample_data)

    assert np.issubdtype(transformed["Gender"].dtype, np.number)

    onehot_cols = [c for c in transformed.columns if c.startswith("City_")]
    assert len(onehot_cols) == len(sample_config["encoding"]["City"]["categories"]) + 1


def test_inverse_transform_roundtrip(sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    transformed = fe.transform(sample_data)
    restored = fe.inverse_transform(transformed)

    np.testing.assert_allclose(restored["Age"], sample_data["Age"], rtol=1e-5)
    np.testing.assert_allclose(restored["Income"], sample_data["Income"], rtol=1e-5)

    expected_categories = set(list(sample_data["Gender"].unique()) + ["<UNK>"])
    assert set(restored["Gender"].unique()) <= expected_categories



def test_save_and_load(tmp_path, sample_config, sample_data):
    fe = FeatureEngineering(sample_config)
    fe.fit(sample_data)
    fe.save(tmp_path)

    loaded = FeatureEngineering.load(tmp_path)
    assert isinstance(loaded, FeatureEngineering)

    assert loaded._config_hash == fe._config_hash
    assert loaded.numerical == fe.numerical

    out = loaded.transform(sample_data)
    assert isinstance(out, pd.DataFrame)
    assert not out.empty


### New code after statical test

def test_customer_care_features():
    """Test the creation of customer care features"""
    df = pd.DataFrame({
        'CustomerCareCalls': [5, 10, 2],
        'RetentionCalls': [2, 4, 1]
    })
    
    result = feature_engineering(df)
    

    assert 'TotalSupportCalls' in result.columns
    assert 'SupportCallsRatio' in result.columns
    
    # check values
    assert result['TotalSupportCalls'].iloc[0] == 7 
    assert result['SupportCallsRatio'].iloc[0] == pytest.approx(5/3, rel=1e-5)  # 5 / (2+1)


def test_change_patterns_features():
    """Test the creation of pattern change features"""
    df = pd.DataFrame({
        'PercChangeMinutes': [10.5, -5.2, 3.0],
        'PercChangeRevenues': [8.3, -3.1, 7.0]
    })
    
    result = feature_engineering(df)
    
    # heck new colonne was created
    assert 'ChangePattern_Combined' in result.columns
    assert 'ChangePattern_Volatility' in result.columns
    
    # Check new values
    expected_combined = (10.5 + 8.3) / 2
    assert result['ChangePattern_Combined'].iloc[0] == pytest.approx(expected_combined, rel=1e-5)
    
    expected_volatility = abs(10.5 - 8.3)
    assert result['ChangePattern_Volatility'].iloc[0] == pytest.approx(expected_volatility, rel=1e-5)


def test_call_behavior_features():
    """Test the creation of call behavior features"""
    df = pd.DataFrame({
        'InboundCalls': [10, 20, 5],
        'OutboundCalls': [5, 10, 2],
        'UnansweredCalls': [2, 3, 1]
    })
    
    result = feature_engineering(df)
    
    #check new columns were created
    assert 'TotalCalls' in result.columns
    assert 'CallDirection_Ratio' in result.columns
    assert 'UnansweredRate' in result.columns
    
    # check new values
    assert result['TotalCalls'].iloc[0] == 15  
    assert result['CallDirection_Ratio'].iloc[0] == pytest.approx(10/6, rel=1e-5)  
    assert result['UnansweredRate'].iloc[0] == pytest.approx(2/16, rel=1e-5)  


def test_retention_features():
    """Test the creation of retention features"""
    df = pd.DataFrame({
        'RetentionOffersAccepted': [2, 1, 0],
        'RetentionCalls': [5, 3, 2]
    })
    
    result = feature_engineering(df)
    
    #check new column was created
    assert 'RetentionAcceptanceRate' in result.columns
    
   #check new values
    assert result['RetentionAcceptanceRate'].iloc[0] == pytest.approx(2/6, rel=1e-5)  # 2 / (5+1)


def test_service_quality_features():
    """Test the creation of service quality features"""
    df = pd.DataFrame({
        'DroppedBlockedCalls': [2, 5, 1],
        'InboundCalls': [10, 20, 5],
        'OutboundCalls': [5, 10, 2]
    })
    
    # First, compute TotalCalls
    result = feature_engineering(df)
    
    
    assert 'ServiceQualityScore' in result.columns
    
    # check new values (1 - (DroppedBlockedCalls / (TotalCalls + 1)))
    total_calls_0 = 10 + 5  
    expected_score_0 = 1 - (2 / (total_calls_0 + 1))
    assert result['ServiceQualityScore'].iloc[0] == pytest.approx(expected_score_0, rel=1e-5)


def test_copy_parameter():
    """Test que le paramètre copy fonctionne correctement"""
    df = pd.DataFrame({
        'CustomerCareCalls': [5, 10, 2],
        'RetentionCalls': [2, 4, 1]
    })
    
    # Test with copy=True (by défault)
    result = feature_engineering(df, copy=True)
    assert id(df) != id(result), "Un nouveau DataFrame devrait être créé"
    assert 'TotalSupportCalls' not in df.columns, "Le DataFrame original ne devrait pas être modifié"


def test_missing_columns():
    """Test quand certaines colonnes sont manquantes"""
    df = pd.DataFrame({
        'CustomerCareCalls': [5, 10, 2]
        # RetentionCalls missing
    })
    
    result = feature_engineering(df)
    
   
    assert 'TotalSupportCalls' not in result.columns
    assert 'SupportCallsRatio' not in result.columns