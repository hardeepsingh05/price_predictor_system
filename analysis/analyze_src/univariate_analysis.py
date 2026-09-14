from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df(pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column/attribute to be analyzed.

        Returns:
        None: This method visualize the distribution of the feature.
        """
        pass

# Concrete Strategy for Numerical Features
# -------------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df(pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column/attribute to be analyzed.

        Returns:
        None: Prints a histogram with KDE plot
        """
        plt.figure(figsize=(12,10))
        sns.histplot(df[feature], kde = True, bins = 30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical featuresby plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical column/feature using a bar plot.

        Parameters:
        df(pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column/attribute to be analyzed.

        Returns:
        None: Plots a bar chart showing the frequency distribution of each category inside column/feature.
        """
        plt.figure(figsize=(12,10))
        sns.countplot(x = feature,data = df,palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation = 45)
        plt.show()


# Context Class that uses a UnivariateAnalysisStategy
# -------------------------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        Strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, feature: str):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        df(pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Execute the Strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)
        
# Example usage
if __name__ == "__main__":


    pass