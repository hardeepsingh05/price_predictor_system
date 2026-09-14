import logging
from abc import ABC, abstractmethod

import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# -------------------------------------------------------
# This class define a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """

        pass

# Concrete Strategy for Log Transformation
# ------------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize their data distribution.
class LogTransformationFeatureEngineering(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initialize the LogTransformation strategy with the specific features to transform.

        Parameters:
        features (list): The list of features for transforming via LogTransformation.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy(deep = True)
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            ) # log1p handles log(0) by calculating log(1+x)
        logging.info(f"Log transformation completed.")
        return df_transformed
    
# Concreate Strategy for Standard Scaling
# ----------------------------------------------
# This strategy applies standard scaling (z-score normalization) to feature, centering their values around zero with unit variance.
class StandardScalingFeatureEngineering(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the Standard Scaling with specific features to scale.

        Parameters:
        features (list): The list of features for transforming via Standard Scaling.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Standard Scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the features to transform.

        Returns: 
        pd.DataFrame: The DataFrame with scaled above specified features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy(deep = True)
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed
    
# Concrete Strategy for Min-Max Scaling
# ----------------------------------------
# This strategy applies Min-Max scaling to input features, scaling to specified range, typically [0,1].
class MinMaxScalingFeatureEngineering(FeatureEngineeringStrategy):
    def __init__(self, features, features_range = (0,1)):
        """
        Initialize the MinMaxScaling with specific features to scale with a specific target range.

        Parameters:
        features (list): The list of features for transforming via MinMxScaling.
        features_range (tuple): The target range for scaling, default is (0,1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=features_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This applies MinMaxScaling to the above specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame is containing features to transform.

        Returns:
        pd.DataFrame: The DataFrame with Min-Max scaled features.
        """
        logging.info(f"Applying Min-Max Scaling to the features: {self.features} with range {self.scaler.feature_range}")
        df_tranformed = df.copy(deep = True)
        df_tranformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("min-Max scaling completed.")
        return df_tranformed
    
# Concrete strategy for One-Hot Encoding
# ----------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them to binary vectors.
class OneHotEncodingFeatureEngineering(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features for tranforming via OneHotEncoding.
        """
        self.features = features
        self.encoder = OneHotEncoder(spare = False, drop = "first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features to transform.

        Returns:
        pd.DataFrame: The DataFrame with encoded features.
        """
        logging.info(f"Applying One-Hot Encoding to features: {self.features}")
        df_transformed = df.copy(deep = True)
        df_encoded = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns = self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns = self.features).reset_index(drop = True)
        df_transformed = pd.concat([df_transformed, df_encoded], axis = 1)
        logging.info("One-Hot encoding completed.")
        return df_transformed
    
# Context Class for Feature Engineering
# -----------------------------------------
# This class uses a FeatureEngineeringStrategy to apply_transformation to a dataset.
class FeatureEngineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineering with a specific strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineering.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be replaced with current strategy for FeatureEngineering.
        """
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strstegy.

        Parameters:
        df (pd.DataFrame): The DataFrame containing features to transform.

        Returns:
        pd.DataFrame: The DataFrame with applied feature engineering transformations.
        """
        logging.info("Appyling feature engineering strategy.")
        return self._strategy.apply_transformation(df)
    
# Example Usage
if __name__ == "__main__":

    pass