import pandas as pd
from typing import Tuple

from src.data_splitter import TrainTestSplitter, SimpleTrainTestSplitStrategy
from zenml import step

@step
def data_splitter_step(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets using DataSplitter and a chosen strategy.
    """

    splitter = TrainTestSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_col)
    return X_train, X_test, y_train, y_test
