import logging 
from abc import ABC, abstractmethod

import numpy as np 
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s -%(message)s")

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data target feature.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """

        pass

# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
            self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
            ) -> dict:
        """
        Evaluate a regression model using R-Squared and Mean Squared Error.

        Parameters:
        model (RegressorMixin): The trained regression Model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.DataFrame): The testing data target feature/label.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error. 
        """

        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    

# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializing the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        
        self._strategy = strategy

    def execute_strategy(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The model to be evaluated.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.DataFrame): The testing data label/target feature.

        Returns:
        dict: A dictionary containing the results of evaluation using different metrics.
        """

        logging.info("Evaluation the model using the seleted strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)
    
# Example Usage 
if __name__ == "__main__":

    pass

        