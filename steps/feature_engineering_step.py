import pandas as pd
import numpy as np 
from src.feature_engineering import (
    FeatureEngineering,
    LogTransformationFeatureEngineering,
    StandardScalingFeatureEngineering,
    OneHotEncodingFeatureEngineering,
    MinMaxScalingFeatureEngineering
)
from zenml import step

@step
def feature_engineering_step(df: pd.DataFrame,features: list = None, strategy: str = "Log Transformation") -> pd.DataFrame:
    """
    Handle feature engineering with given specific strategy and features of the DataFrame.
    """
    # Ensire features is a list, even if not provided
    if features is None:
        raise ValueError (f"Features list is not provided.")
    
    if strategy == "Log Transformtion":
        feature_engineer = FeatureEngineering(LogTransformationFeatureEngineering(features))
    elif strategy == "Standard Scaling":
        feature_engineer.set_strategy(StandardScalingFeatureEngineering(features))
    elif strategy == "One-Hot Encoding":
        feature_engineer.set_strategy(OneHotEncodingFeatureEngineering(features))
    elif strategy == "Min-Max Scaling":
        feature_engineer.set_strategy(MinMaxScalingFeatureEngineering(features))
    else:
        raise ValueError (f"Unsupported feature engineering technique: {strategy}")

    feature_engineered_df = feature_engineer.execute_strategy(df)
    return feature_engineered_df
