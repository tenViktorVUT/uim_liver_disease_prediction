#%%%

#TODO: Remove prints and add log messages
#TODO: OSEKAŤ NEFYZIOLOGICKÉ HODNOTY
#TODO: ZJEDNOTIŤ DOKUMENTÁCIU
# TODO: MODEL A VÝBER HYPERPARAMETROV

"""
LIVER DISEASE PREDICTION

This project was created for BPC-UIM (Umělá inteligence v medícíne) class @ VUT Brno.

Created by
Viktor Morovič
VUT: 257026@vutbr.cz
 
Filip Sedlár
VUT: 
 
Matúš Smolka
VUT: 257044@vutbr.cz
"""

# importing dependencies
# built-in libs
import os
import time
import logging

# NN
import tqdm
import shap

# Basic data analytics libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Principal component analysis
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    f1_score
)

from sklearn.decomposition import PCA
from torch.utils.hipify.hipify_python import preprocessor

# Classificator XGBoost
# from xgboost import XGBClassifier

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

#%%%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)


def load_file(filename: str) -> pd.DataFrame:
    """
    Loads CSV data under filename into 2 pandas DataFrames
    :param (str) filename: name of the file
    :return:
        rdf - raw DataFrame
        df - deep copy DataFrame
    """

    try:
        # Finding path
        cwd = os.getcwd()
        path = os.path.join(cwd,filename)
        rdf = pd.read_csv(path)
        
        # Creates deep copy of df
        df = rdf.copy(deep=True)
        logger.info(f'File {filename} loaded succesfully. \n')
        
        # print(f'Dataset obsahuje {df.shape[0]} řádků a {df.shape[1]} sloupců.')
        # TEST PRINT
        # print(df.head()) 
        return rdf, df
   
    except FileNotFoundError:
        logger.info(f'File {path} was not found in directory.')
        return None

#%%%
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses data into a clean working DataFrame.
    Takes in raw data frame and replaces non-sense values with
    NaNs
    :param (pd.DataFrame) df: unprocessed DataFrame
    """
    
    ### REFORMATTING SELECTOR OUTPUT INTO BINARY VALUES
    if 'Selector' not in df.columns:
        # raises error in case of missing selector column
        raise NameError("Selector column missing in the DataFrame")
    
    # mapping binary values onto selector column
    # 1: pathological 2->0:healthy
    df['Selector'] = df['Selector'].map({1:1,2:0})
    
    ### REPLACING NON-SENSE AGE
    df.loc[df['Age'] > 110, 'Age'] = np.nan
    
    ### REFORMATTING GENDER INTO BINARY, DISCRETE VALUES
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
    
    ### CORRECTING NEGATIVE VALUES
    df[df<0] = np.nan
    
    return df


def del_missing(df:pd.DataFrame) -> pd.DataFrame:
    """
    Removes entries with missing Selector value - NaN
    :param (pd.DataFrame) df: DataFrame
    :return:
        df - DataFrame with removed missing Selector entries 
    """

    logger.info('Removing entries with missing Selector...')
    raw_count = len(df)
    df.dropna(subset='Selector', inplace=True)
    new_count = len(df)
    deleted = raw_count - new_count
    if deleted > 0:
        logger.info(f'Removed {deleted} entries with missing Selector')
    else:
        logger.info('No missing entries without Selector in the DataFrame')

    logger.info(f'Current number of entries in dataset: {new_count}')
    return df

def graph_data(df: pd.DataFrame) -> None:
    """
    Function for unifying graphing functions under one function
    :param (pd.DataFrame) df: DataFrame
    :return: None     
    """
    def graph_shape(df:pd.DataFrame) -> None:
        """
        Plots every feature from a DataFrame df. 
        New figure is created after closing the previous one. 
        :param df: DataFrame
        :return:
            None
        """
        
        # Selects every numeric column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Iterates over columns and for each one generates a histogram
        for col in numeric_cols:
            plt.figure()
            df[col].hist(bins=30)
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
            
        return None

    def get_corelation_matrix(df:pd.DataFrame) -> None:
        """
        Generates correlation matrix and plots it in a heatmap.
        :param (pd.DataFrame) df: DataFrame
        :return:
            None
        """
        corr = df.corr(numeric_only=True)

        # Plotting of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.title("Correlation Matrix of Biomarkers")
        plt.show()
        return None

    def plot_gender(df:pd.DataFrame) -> None:
        """
        Plots gender with hue showing Selector
        :param (pd.DataFrame) df: DataFrame
        :return: None
        """
        # Vizualizace rozdělení pohlaví
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x='Gender',
            discrete=True,
            hue='Selector',
            palette='rocket',
            shrink=.8).set_xticks([0, 1])
        plt.title('Rozdělení pacientů podle pohlaví')
        plt.xlabel('Pohlaví (0=Muž, 1=Žena)')
        plt.ylabel('Počet')
        plt.show()
        return None
        
    # Function calling
    logger.info('Visualising data...')
    get_corelation_matrix(df=df)
    graph_shape(df=df)
    plot_gender(df=df)
    
    return None

def fill_miss_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in missing values (NaNs) using KNNImputer for continuous data
    and most common value (modus) for discrete categoricacl data. 
    :param (pd.DataFrame) df: DataFrame of preprocessed data
    :return:
        df_imputed - DataFrame s doplněnými hodnotami
    """

    logger.info('Filling in missing values...')
    # Oddělení hodnoty kterou nechci upravovat
    selector_col = False
    if 'Selector' in df.columns:
        df_target = df['Selector'].copy(deep=True)
        df_features = df.drop('Selector', axis=1)
        selector_col = True
    else:
        df_features = df

    # Rozdělení sloupců na numerická a kategorické
    categorical_features = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df_features.select_dtypes(include=['number']).columns.tolist()

    logger.info(f'Found {len(numerical_features)} numerical features.\n')
    logger.info(f'Found {len(categorical_features)} categorical features.\n')

    # Pipelines
    numerical_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=5))
    ]) # 5 Sousedů
    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])

    # Kombinace transformací
    preproc = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
        )

    # Pozor, preprocessor vrací NumPy
    df_imputed_array = preproc.fit_transform(df_features)
    df_imputed = pd.DataFrame(
        df_imputed_array,
        columns=df_features.columns,
        index=df_features.index
        )

    # Připojení 'Selector'
    if selector_col:
        df_imputed['Selector'] = df_target

    logger.info('Doplňování chybějících hodnot dokončeno.')

    return df_imputed


def split_data(
    data: pd.DataFrame,
    seed:int = 42
    ) -> pd.DataFrame:
    """
    Splits data into training and validation data
    
    :param (pd.DataFrame) data: original data
    :return train_x: training data
    :return train_y: training for error
    :return val_x: validation data
    :return val_y: validation error
    """
    
    # Getting all columns names
    features = data.columns
    # Removing our Y from the list -> Selector
    features = [col for col in features if col != 'Selector']
    
    # Assigning X, Y
    Y = data.Selector
    X = data[features]
    (
        train_x, val_x, train_y, val_y
        ) = train_test_split(
            X, Y,
            test_size=0.8,
            random_state=seed
            )
        
    return train_x, val_x, train_y, val_y

#%%%
# --- Hlavní skript ---
if __name__ == "__main__":

    logger.info('Loading data file...')
    # Načtení souboru
    path = 'liver-disease_data.csv'
    rdf, df = load_file(path)
    # display(df)

    if df is not None:

        df = preprocess_data(df=df)
        # display(df)

        # Odstranění řádků s chybějící cílovou hodnotou
        df = del_missing(df)
        display(df)
#%%%
        # Doplnění chybějících hodnot
        df = fill_miss_values(df)

        logger.info('Data preprocessing completed')
        # print('Počet chybějících hodnot (NaN) v každém sloupci po základním zpracování:')
        # print(df.isnull().sum()) # Správně nuly...


    #%%%
        graph_data(df=df)
#%%%
display(df)




