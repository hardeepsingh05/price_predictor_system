import pandas as pd
import logging
from src.outlier_detection import (
    ZScoreOutlierDetection,
    IQROutliersDetection,
    OutlierDetector
)
from zenml import step

@step
def outlier_detection_step(df: pd.DataFrame, strategy: str, column_name: str) -> pd.DataFrame:
    """
    Outlier detection, handling and visualization using OutlierDetector.
    """
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.info("Received a NoneType DataFrame.")
        raise ValueError("Input dataframe must be a non-null pandas DataFrame.")
    
    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input dataframe must be a pandas DataFrame.")
    
    if column_name not in df.columns:
        logging.error(f"Column {column_name}' does not exist in the DataFrame")
        raise ValueError(f"Column '{column_name}' does not  exist in the DataFrame.")

    if strategy == "ZScore":
        outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    elif strategy == "IQR":
        outlier_detector.set_strategy(IQROutliersDetection())
    else:
        logging.error(f"Unknown outlier detection strategy: {strategy}.")
        raise ValueError(f"Unknown outlier detection strategy: {strategy}.")
    
    # Filtering numeric columns from all others from the DataFrame for hnadling outliers
    df_numeric = df.select_dtypes(include=[int,float])   
    ouliers = outlier_detector.detect_outlier(df_numeric)
    df_cleaned = outlier_detector.handle_outlier(df_numeric, method="remove")
    return df_cleaned