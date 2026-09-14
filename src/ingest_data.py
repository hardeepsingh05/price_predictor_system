import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Define an abstract class for Data Ingestor
class Data_Ingestor(ABC):
    @abstractmethod
    def Ingest(self,file_path: str) -> pd.DataFrame:
        # Abstract method to ingest data from a given file.
        pass

# Implement a concrete class for ZIP file Ingestion
class ZipData_Ingestor(Data_Ingestor):
    def Ingest(self,file_path: str) -> pd.DataFrame:
        """Extract a .zip file and returns the content as a pandas DataFrame"""
        # Ensuring the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("provided file is not a .zip file")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path,"r") as zp_file:
            zp_file.extractall("extracted_data")

        # Find the extracted CSV file (let's find out is there any CSV file got extracted in folder)
        extracted_files = os.listdir("extracted_data")
        csv_files = [file for file in extracted_files if file.endswith(".csv")]

        if len(csv_files) == 0: 
            raise FileNotFoundError("No CSV file found in the extracted data")
        if len(csv_files) > 1: 
            raise ValueError("Multiple CSV files found. Please specify which one to use")
        
        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df        

# Implement a Factory to create DataIngestors and call for different file format based on availability
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> Data_Ingestor:
        """Return the appropriate DataIngestor based on file extension"""
        if file_extension == ".zip":
            return ZipData_Ingestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")
        

# Example usage:
if __name__ == "__main__":
    # Specify file_path
    # file_path = "C:/Users/harde/OneDrive/Documents/Learning Data Science/MLOPS/Price_Predictor_System/data/archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropraite DataIngestor
    # data_Ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load it into a DataFrame
    # df = data_Ingestor.Ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    # print(df.head())   # Display the first few rows from DataFrame

    pass