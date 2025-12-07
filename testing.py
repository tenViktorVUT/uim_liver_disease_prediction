"""
THIS SCRIPT SERVES FOR VALIDATION DATASET ONLY

This project was created for BPC-UIM (Umělá inteligence v medícíne) class @ VUT Brno.

REQUIREMENTS:   
Existing model.pkl in the current working directory,
Existing external validation dataset


Created by
Viktor Morovič
VUT: 257026@vutbr.cz

Filip Sedlár
VUT: 262751@vutbr.cz

Matúš Smolka
VUT: 257044@vutbr.cz
"""

# Importing libraries
import os
import time
import logging
import joblib
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from main import preprocess_data, del_missing

# Setting up logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """"
    Loads dataset and reads it as pandas DataFrame
    
    :param None:
    :return pd.DataFrame: 
    """
    
    # Getting current working directory and joining path 
    cwd = os.getcwd()
    
    print()
    filename = input("\033[1mInput csv filename of validation data \033[0m\nCSV file must be in the current working directory: ")
    
    # Basic input cleaning
    filename = filename.lower().strip()
    path = os.path.join(cwd, filename)
    
    # Reading data
    df =  pd.read_csv(path)
    logger.info("Validation data loaded succesfully...")
    
    return df


if __name__ == "__main__":
    # loading in dataset
    df_raw = load_dataset()
    
    # preprocessing data
    time.sleep(1)
    logger.info("Preprocessing data...")
    df = preprocess_data(df=df_raw)
    df = del_missing(df=df)
    time.sleep(1)
    logger.info("Data preprocessing completed...")
    
    # Extracting df into features and target
    y_true = df['Selector']
    X = df.drop(columns=['Selector'])
    
    # Loading in pre-trained model
    time.sleep(1)
    logger.info("Loading model...")
    model = joblib.load('model.pkl')
    time.sleep(1)
    logger.info("Model loaded succesfully...")

    # Model prediction
    y_pred = model.predict(X)
    
    # Calculating MCC
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    
    print("\nTESTING MCC:")
    print(mcc)
    
    # This is the end of the script :)
    

