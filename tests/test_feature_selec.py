import numpy as np
import pandas as pd
from pipelines.feature_selection_utils import identify_features_to_drop, identify_and_drop

def test_identify_and_drop_basic():
    np.random.seed(0)
    df = pd.DataFrame({
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "feature3": np.random.rand(10),
        "HandsetPrice": [np.nan if i % 3 == 0 else np.random.rand() for i in range(10)]
    })
    X_train_preprocess = df.copy()

    iv_scores_series = pd.Series({
        "feature1": 0.00005,
        "feature2": 0.2,
        "feature3": 0.00001,
        "HandsetPrice": 0.15
    })
    mutual_info = pd.Series({
        "feature1": 0.0005,
        "feature2": 0.002,
        "feature3": 0.0001,
        "HandsetPrice": 0.005
    })

    to_drop = identify_features_to_drop(df, X_train_preprocess, iv_scores_series, mutual_info)
    assert isinstance(to_drop, list)
    assert "HandsetPrice" in to_drop

    df_after, dropped = identify_and_drop(df, X_train_preprocess, iv_scores_series, mutual_info, inplace=False)
    assert "HandsetPrice" not in df_after.columns
    assert set(dropped).issubset(set(["feature1", "feature3", "HandsetPrice", "feature2"]))

