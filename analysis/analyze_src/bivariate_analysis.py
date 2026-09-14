from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame,feature01: str,feature02: str):
        """
        Perform bivariate analysis on two features of the dataframe

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature01 (str): The name of the first column/feature to be analyzed.
        feature02 (str): The name of the second column/feature ot be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass

# Concrete Strategy for Numerical vs Numercial Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plot.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature01: str, feature02: str):
        """
        Plot the relationship between the two numercial features using scatter plot.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature01 (str): The name of the first column/feature to be analyzed.
        feature02 (str): The name of the second column/feature ot be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between two numerical features.
        """

        plt.figure(figsize=(12,10))
        sns.scatterplot(x = feature01, y = feature02, data = df)
        plt.title(f"{feature01} vs {feature02}")
        plt.xlabel(feature01)
        plt.ylabel(feature02)
        plt.show()

# Concreate Strategy for Categorical vs Numerical Analysis
# -----------------------------------------------------------
# This strategy analyzes the relationship between a categorical and numercial feature using box plot.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature01: str, feature02: str):
        """
        Plots the relationship between a categorical and numerical feature using box plot.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature01 (str): The name of the first column/feature to be analyzed.
        feature02 (str): The name of the second column/feature ot be analyzed.

        Returns:
        None: Displays a boc plot showing the relationship between the numerical and categorical features.
        """

        plt.figure(figsize = (12,10))
        sns.boxplot(x = feature01, y = feature02, data = df)
        plt.title(f"{feature01} vs {feature02}")
        plt.xlabel(feature01)
        plt.ylabel(feature02)
        plt.xticks(rotation = 45)
        plt.show()

# Context Class that uses a BivariateAnalysisStrategy
# ------------------------------------------------------
# This class allows you to switch between different bivariate analysis strategies
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None 
        """

        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None 
        """

        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, feature01: str, features02: str):
        """
        Execute the bivariate analysis between given columns/feartures using current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature01 (str): The name of first feature/column to be analyzed.
        feature02 (str): The name of second feature/column to be analyzed.

        Returns:
        None: Executes the current bivariate analysis strategy for given two features/columns/attributes.
        """
        self._strategy.analyze(df,feature01,features02)

# Example usage
if __name__ == "__main__":

    pass



        