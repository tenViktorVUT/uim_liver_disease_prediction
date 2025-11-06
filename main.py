#%%%

#FIXME: FILL MISSING REMOVES SELECTOR COLUMN
#TODO: TRAIN_TEST_SPLIT
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
VUT: 262751@vutbr.cz
 
Matúš Smolka
VUT: 257044@vutbr.cz
"""

# importing dependencies
# built-in libs
import os
import time


# NN
import tqdm
import shap


# Basic data analytics libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        logger.warning(f'File {path} was not found in directory.')
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


def del_missing(df):
    """
    Deletes the row, if there is a 'Selector' value missing (NaN)
    :param df: DataFrame
    :return:
        df - DataFrame with deleted NaN 'Selector' values
    """

    logger.info('Checking the values in the "Selector"...')
    raw_count = len(df)
    df.dropna(subset='Selector', inplace=True)
    new_count = len(df)
    deleted = raw_count - new_count
    if deleted > 0:
        logger.info(f'Deleted {deleted} rows, where there was an error in "Selector" value.')
    else:
        logger.info('There were no missing values in "Selector".')

    logger.info(f'Current number of rows in the dataset: {new_count}')
    return df

def graph_shape(df):
    """
    Stĺpec po stĺpci vytvorý grafy pre každý parameter aby sme mohly určiť rozloženie.
    Grafi sa zobrazujú jeden podruho vždy až po zatvorení predšlého grafu
    :param df: DataFrame
    :return:
        None
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    return None

def get_corelation_matrix(df):
    """
    vytvorí maticu korelácí jednotlivých parametrou pre určenie miery korelácie a potencionalne odhalenie redundancie .
    :param df: DataFrame
    :return:
        None
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Matrix of Biomarkers")
    plt.show()
    return None


def fill_miss_values(df):
    """
    Fillip up the missing values (NaN) with the help of KNNImputer for numerical values
    and with modus for categorical.
    :param df: DataFrame
    :return:
        final_df - DataFrame with filled values, Selector without change
    """

    logger.info('Working on filling the NaN values...')
    df = df.copy()

    # Separating the feature I do not want to change
    selector_col = None
    if 'Selector' in df.columns:
        selector_col = df['Selector'].copy()
        df_features = df.drop(columns=['Selector'])
    else:
        df_features = df

    # Differentiate between numerical and categorical
     # Just failsafe, we did change gender into binary form.
    categorical_features = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df_features.select_dtypes(include=['number']).columns.tolist()

    logger.info(f'Found {len(numerical_features)} numerical values.\n')
    logger.info(f'Found {len(categorical_features)} categorical values.\n')

    # Pipelines
    transformers = []
    if numerical_features:
        numerical_transformer = Pipeline(steps=[
            ('impute', KNNImputer(n_neighbors=5))
        ])  # 5 Neighbours
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        # No need for transformation
        if selector_col is not None:
            final_df = pd.concat([df_features, selector_col], axis=1)
        else:
            final_df = df_features
            logger.info('No columns for imputation, returning original DataFrame.\n')
        return final_df

    # Combination of transformations
    preproc = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough')

    # Fit-transform
    imputed_array = preproc.fit_transform(df_features)

    # Order of returned columns
        # ! ColumnTransformer returns values based on order of transformers
    transformed_cols = []
    for name, transformer, cols in preproc.transformers_:
        if name != 'remained':
            transformed_cols.extend(cols)
    # Remained columns if they exist
    if preproc.remainder == 'passthrough':
        passthrough_cols = [c for c in df_features.columns if c not in transformed_cols]
        transformed_cols.extend(passthrough_cols)

    # Create a DataFrame and add index
    df_imputed = pd.DataFrame(imputed_array, columns=transformed_cols, index=df_features.index)

    # Return original data types
    for col in categorical_features:
        if col in df_imputed.columns:
            orig_dtype = df_features[col].dtype
            if pd.api.types.is_categorical_dtype(orig_dtype):
                df_imputed[col] = df_imputed[col].astype('category')
            else:
                df_imputed[col] = df_imputed[col].astype(df_features[col].dtype)

    # Checking the original column order
    df_imputed = df_imputed[df_features.columns]

    # Adding back the 'Selector'
    if selector_col is not None:
        final_df = pd.concat([df_imputed, selector_col], axis=1)
    else:
        final_df = df_imputed

    logger.info('Correcting of the missing values was finished.')

    return final_df


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

#%%%
# --- Hlavní skript ---
if __name__ == "__main__":

    logger.info('\nLoading of the file in progress...')
    # Načtení souboru
    path = 'liver-disease_data.csv'
    rdf, df = load_file(path)
    #display(df)

#%%%
    if df is not None:

        df = preprocess_data(df=df)
        #display(df)

#%%%
        # Odstranění řádků s chybějící cílovou hodnotou
        df = del_missing(df)
        #display(df)
#%%%
        # Doplnění chybějících hodnot
        df = fill_miss_values(df)

        print('\nPreprocessing finished.')
        print('Number of missing values (NaN) after basic preprocessing:')
        print(df.isnull().sum()) # Správně nuly...

        # Vizualizace rozdělení pohlaví
        print('Creating the visualisation for gender distribution...')
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x='Gender',
            discrete=True,
            shrink=.8).set_xticks([0, 1])
        plt.title('Rozdělení pacientů podle pohlaví')
        plt.xlabel('Pohlaví (0=Muž, 1=Žena)')
        plt.ylabel('Počet')
        plt.show()

#%%%
print(df.head)




