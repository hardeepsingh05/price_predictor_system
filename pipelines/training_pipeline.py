from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from zenml import Model, pipeline, step

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv() 

# Define pipeline name
@pipeline(
    model = Model(
        # The name would be used to uniquely identifies this model
        name = "prices_predictor"
    ),
)

def ml_pipeline():
    """
    Define an end-to-end machine learning pipeline.
    """

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path =  os.getenv("INPUT_DATA_FILE_PATH")
    )

    # handling Missing Vlaues Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering
    engineered_data = feature_engineering_step(
        filled_data, features = ["Gr Liv Area", "SalePrice"],strategy = "Log Transformation"
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, strategy = "ZScore", column_name = "SalePrice")

    # Train-Test-Split of  DataFrame
    X_train,X_test, y_train, y_test = data_splitter_step(df = clean_data, target_col = "SalePrice")
        
    # Model Building Step
    model = model_building_step(X_train = X_train, y_train = y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model = model, X_test = X_test, y_test = y_test
    )

    return model

# Example Usage
if __name__ == "__main__":
    # Running the pipelines
    # run = ml_pipeline()

    pass