import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from pipelines.feature_selection import feature_selection
from pipelines.feature_engineering import feature_engineering


# preprocessing
preprocessor = make_column_transformer(
    (make_pipeline(
        FunctionTransformer(),
        SimpleImputer(strategy='median'),
        RobustScaler()
    ), make_column_selector(dtype_include=['float64', 'int64'])),
    (make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ), make_column_selector(dtype_include=['object', 'category'])),
    verbose_feature_names_out=False
)

preprocessor.set_output(transform="pandas")


def pipeline(df: pd.DataFrame, copy: bool = True, train: bool = True) -> pd.DataFrame:
    """
        Complete data preprocessing pipeline.
        
        Args:
            df: DataFrame with the data
            copy: If True, create a copy of the DataFrame
            train: If True, fit the preprocessor, otherwise only transform
            
        Returns:
            Transformed DataFrame
    """
    if copy:
        df = df.copy()

    df = feature_selection(df, copy=False)
    df = feature_engineering(df, copy=False)

    if train:
        df = preprocessor.fit_transform(df)
    else:
        df = preprocessor.transform(df)

    return df