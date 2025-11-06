"""
LIVER DISEASE PREDICTION

This project was created for BPC-UIM (Umělá inteligence v medícíne) class @ VUT Brno.

Created by
Viktor Morovič
VUT: 257026@vutbr.cz
 
Filip Sedlár
VUT: 
 
Matúš Smolka
VUT: 
"""

# importing dependencies
# built-in libs
import os
import time

# NN
import tqdm
import shap

# basic data analytics libs
import numpy as np
import pandas as pd
import seaborn as sns

# Principal component analysis
from sklearn.decomposition import PCA

# pozrieť jednotlivé scipy moduly pre rýchlejšie načítanie


"""
Features explanation: <br>
- Věk pacienta (Age of the patient) <br>
- Pohlaví pacienta (Gender of the patient)<br>
- Celkový bilirubin (Total Bilirubin)<br>
- Přímý bilirubin (Direct Bilirubin)<br>
- Alkalická fosfatáza (Alkaline Phosphatase)<br>
- Alaninaminotransferáza (Alamine Aminotransferase, ALT)<br>
- Aspartátaminotransferáza (Aspartate Aminotransferase, AST)<br>
- Celkové bílkoviny (Total Proteins)<br>
- Albumin (Albumin)<br>
- Poměr albumin/globulin (Albumin and Globulin Ratio)<br>
- Dataset: Pole určující, zda pacient spadá do skupiny s onemocněním jater nebo bez něj<br>

***classification*** - patient is sick / healthy
"""

# Loading data and analysis
def load_data(filename:str) -> pd.DataFrame:
    """
    Loads csv data into a pandas Dataframe
    
    :return pd.Dataframe:
    """
    
    # Geetting path to load data
    path = os.getcwd()
    
    # loading raw DataFrame
    rdf = pd.read_csv(f"{path}/{filename}")
    # Creating deep copy and replacing negative (non-sense) values
    df = rdf.copy(deep=True)


    def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function for cleaning strong outliers in Dataframe
        
        :param (pd.DataFrame) raw: raw unprocessed DataFrame
        :return pd.DataFrame:  pre-processed DataFrame  
        """
        
        # Assigning gender discrete values 
        _ = {"Male": 0, "Female": 1}
        df['Gender'] = df['Gender'].replace(_)
        
        # Replacing all negative values with NaN
        df[df < 0] = np.nan
        
        return df
    
    
    df = clean_data(raw=df)
    
    return df


def split_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Splits data into training and validation data
    
    :param (pd.DataFrame) data: original data
    :return train_x: training data
    :return train_y: training for error
    :return val_x: validation data
    :return val_y: validation error
    """
    
    # pridať train_test_split
    pass



if __name__ == "__main__":
    # Testing functions and algo here
    load_data("liver-disease_data.csv")