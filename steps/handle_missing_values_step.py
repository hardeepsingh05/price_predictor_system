import pandas as pd
from src.handle_missing_values import (
    MissingValuesHandler,
    FillingMissingValuesStrategy,
    DropMissingValuesStrategy
)
from zenml import step

@step
def handle_missing_values_step(df: pd.DataFrame,strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values using MissingValuesHandler with the specified strategy.
    """

    if strategy == "drop":
        data_handler = MissingValuesHandler(DropMissingValuesStrategy(axis = 0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        data_handler = MissingValuesHandler(FillingMissingValuesStrategy(method = strategy))
    else:
        raise ValueError (f"Unsupported missing values handling strategy: {strategy}")
    
    cleaned_df = data_handler.handle_missing_values(df)
    return cleaned_df