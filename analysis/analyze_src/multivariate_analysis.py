from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

# Abstract Base Class for Multivariate Analysis
# ---------------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive analysis by genreating a correlation heatmap and pair plot.

        Parameters: 
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features.

        Parameters:
        df (pd.DataFrame): The dataframe containing data to be analyzed.

        Returns:
        None: This method should generate and display a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a pair plot.
        """
        
        pass

# Concreate class for Multivariate Analysis with Correlation Heatmap and Pair Plot
# --------------------------------------------------------------------------------
# This class implements the method to generate a correlation heatmap and a pair plot.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlation between numerical columns/features.
        """

        plt.figure(figsize = (12,10))
        sns.heatmap(df.corr(), annot = True, fmt = ".2f", cmap = "coolwarm", linewidths = 0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features in the dataframe

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features",y = 1.02)
        plt.show()

# Example Usage
if __name__ == "__main__":

    pass
        
        