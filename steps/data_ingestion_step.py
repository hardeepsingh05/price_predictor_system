import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Ingest data from a ZIP file using the appropriate DataIngestor.
    """
    # Determine the file extension
    file_extension = ".zip"

    # Get the apporpriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.Ingest(file_path)

    # Return ingested dataframe
    return df
