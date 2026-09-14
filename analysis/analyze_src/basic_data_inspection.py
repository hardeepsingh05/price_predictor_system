from abc import ABC, abstractmethod

import pandas as pd

# Abstract Base Class for Data Inspection Strategies
# ------------------------------------------------------------------
# This class defines a common interface for data inspection strategies
# SubClasses must implement the inspect method

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of Data inspection.

        Parameters: 
        df(pd.DataFrame): The dataframe on which the inspection to be performed

        Returns:
        None: This method prints the inspection results directly
        """

        pass

# Concrete Strategy for Data Type Inspection
# ----------------------------------------------------------------------------------
# This strategy inspects the data type of each columns and count of non-value values
class DataTypeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of values from each column of input dataframe

        Parameters:
        df (pd.DataFrame): the dataframe to be inspected

        Returns:
        None: Prints the data types and non-null counts to the columns
        """

        print("\n Data Types and Non-Null Counts by Columns are as follow: \n")
        print(df.info())

# Concrete Stategy for Summary Statistics Inspection
# ------------------------------------------------------------
# This strategy provides summary statistics from both numberical and categorical features
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features

        Parameters:
        df (pd.DataFrame): the dataframe to be inspected

        Returns:
        None: Prints summary statistics for data features
        """
        print("\n Summary Statistics (Numerical Features): ")
        print(df.describe())
        print("\n Summary Statistics (Categorical Features): ")
        print(df.describe(include=['O']))

# Concrete Strategy for shape and size Inspection
# ----------------------------------------------------------------------------------
# This strategy inspects the number of rows and columns in dataframe
class ShapeSizeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the number of rows and columns in dataframe

        Parameters:
        df (pd.DataFrame): the dataframe to be inspected

        Returns:
        None: Prints the format: (number of rows, number of columns) in dataframe
        """

        print("\n Number of Rows and Columns in data are as follow: \n")
        print(f"Number of Rows: {df.shape[0]}, Number of Columns: {df.shape[1]}")



# Concrete strategy that uses a DataInspectionStrategy
# -------------------------------------------------------
# This class allows you to switch between different data inspection strategies
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy

        Parameters:
        strategy(DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self.inspection_strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector

        Parameters: 
        strategy(DataInspectionStrategy): The new strategy to be used for data inspection

        Returns: 
        None
        """
        self.inspection_strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current(user given) strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Execute the strategy's inspection method.
        """
        self.inspection_strategy.inspect(df)


# Example Usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies

    # Load the data
    df = pd.read_csv("C:/Users/harde/OneDrive/Documents/Learning Data Science/MLOPS/Price_Predictor_System/extracted_data/AmesHousing.csv")

    # Initialize the Data Inspector with a specific strategy
    inspector = DataInspector(DataTypeInspectionStrategy())
    inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and Execute
    inspector = DataInspector(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)

     # Change strategy to Shape&Size and Execute
    inspector.set_strategy(ShapeSizeInspectionStrategy())
    inspector.execute_inspection(df)

    print("\n******basic_data_inspection.py file has been executed*************\n")
    pass

    