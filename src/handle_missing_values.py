import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")

# Abstract base Class for Missing Value handling Strategy
class MissingValuesHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame

        Parameters:
        df (pd.DataFrame); The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with all missing values handled.
        """

        pass

# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init(self, axis = 0, thresh = None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values and 1 to drop columns with missing values.
        thresh (int): The threshold for number of non-missing values. Rows/Columns having minimum threshold non-missing values should not be dropped.
        """

        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing rows and columns handled.
        """

        logging.info(f"Dropping missing values with axis = {self.axis} and threshold = {self.thresh}")
        df_cleaned = df.dropna(axis = self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned
    

# Concrete Strategy for Filling Missing Values
class FillingMissingValuesStrategy(MissingValuesHandlingStrategy):
    def __init__(self, method = "mean", fill_value = None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (str): The constant value to fill missing value when method = 'constant'.
        """

        self.method = method
        self.fill_value = fill_value 
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or with constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled. 
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include = "number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include = "number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(
                    df[column].mode().iloc[0], inplace = True
                )
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.info(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned
    
# Context Class for handling Missing Values
class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing values handling strategy.

        Parameters:
        stretegy (MissingValuesHandlingStrategy): The strategy to be used to handle missing values in the DataFrame
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        """
        Sets a new strategy to handle missing values for MissingValuesHandler.

        Paramters:
        set_strategy (MissingValuesHandlingStrategy): The new strategy to handle missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame}: The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame missing values handled.
        """
        logging.info("Executing missing value handling strategy with current strategy.")
        return self._strategy.handle(df)
                     

# Example usage
if __name__ == "__main__":

    pass