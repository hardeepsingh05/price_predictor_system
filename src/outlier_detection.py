import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

# Setup logging configuration 
logging.basicConfig(level = logging.INFO, format =  "%(asctime)s - %9levelname)s - % (message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features with outliers.

        Returns:
        pd.DataFrame: The boolean DataFrame indicating where outlier are located. 
        """

        pass

# Concrete Strategy for z-score Based outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold = 3):
        self.threshold = threshold

    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using Z-Score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"outliers detected with Z-Score threshold: {self.threshold}.")
        return outliers
    
# Concrete strategy for IQR(Inter-Quartile Range) Based Outlier Detection
class IQROutliersDetection(OutlierDetectionStrategy):
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - (1.5 * IQR)
        upper_fence = Q3 + (1.5 * IQR)
        outliers = (df < lower_fence) | (df > upper_fence)
        logging.info("Outliers detected using the IQR method.")
        return outliers
    
# Context Class for Outliers Detection and Handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing outliers detection strategy.")
        return self._strategy.detect_outlier(df)
        
    def handle_outlier(self, df: pd.DataFrame, method = "remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outlier(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower = df.quantile(0.01), upper = df.quantile(0.99), axis = 1)
        else:
            logging.info(f"Unknown method: '{method}'. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize =(12,10))
            sns.boxplot(x = df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outliers visualization completed.")

# Example Usage
if __name__ == "___main__":

    pass
