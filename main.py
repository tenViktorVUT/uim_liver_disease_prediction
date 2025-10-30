"""
LIVER DISEASE PREDICTION

This project was created for BPC-UIM (Umělá inteligence v medícíne) class @ VUT Brno.

Created by
Viktor Morovič
Filip Sedlár
Matúš Smolka
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

# Classificator XGBoost
from xgboost import XGBClassifier

# pozrieť jednotlivé scipy moduly pre rýchlejšie načítanie

#Loading data and analysis

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


def load_file(path):
    """
    Načtení dat z CSV souboru do pandas DataFrame
    :param file: CSV soubor k načtení
    :return:
        rdf - pandas raw DataFrame
        df - pandas kopie DataFrame
    """

    # Nalezení cesty, otevírání souboru, oznámení nenačtení

    try:
        #path = os.getcwd()
        #rdf = pd.read_csv(f"{path}/liver-disease_data.csv")
        rdf = pd.read_csv(path)
        df = rdf.copy(deep=True)
        print(f'Soubor {path} byl úspěšně načtený. \n')
        print(f'Dataset obsahuje {df.shape[0]} řádků a {df.shape[1]} sloupců.')
        return rdf, df
        # TEST PRINT
        # print(df.head())
    except FileNotFoundError:
        print(f'Datový soubor {path} nebyl nalezen.')
        return None


def reformat_final(df):
    """
    Překóduje 'Selector' na binární formát 0 a 1
    Původní: 1 -> 1 (nemocný), 2 -> 0 (zdravý)
    :param df: DataFrame
    :return:
        df - DataFrame s upraveným 'Selector'
    """
    print('\n Probíhá překódování sloupce "Selector"...')
    if 'Selector' not in df.columns:
        print('Pozor! Sloupec "Selector" nebyl nalezen.')
        return df

    print('Původní unikátní hodnoty Selector:', df['Selector'].unique())

    mapping = {1: 1, 2:0}
    df['Selector'] = df['Selector'].map(mapping)

    print(f'Sloupec "Selector" byl překódován. Nové unikátní hodnoty:', df['Selector'].unique())

    return df


def negativ_num_correct(df, column_name):
    """
    Identifikace a opravení záporných hodnot v sloupci.
    Záporné hodnoty jsou nahrazeny NaN.
    Vypsání počtu nalezených a pak opravených hodnot.
    :param df: DataFrame, v kterém se bude konat oprava
    :param column_name: Název sloupce k kontrole a opravě
    :return: DataFrame s opravenými hodnotami
    """

    num_of_negativ = (df[column_name]<0).sum()

    if num_of_negativ > 0:
        # Nahrazení hodnot za NaN
        df.loc[df[column_name] < 0, column_name] = np.nan
        print(f'V sloupci {column_name} bylo nalezeno a opraveno {num_of_negativ} záporných hodnot. \n ')
    else:
        print(f'V sloupci {column_name} nebyly nalezeny žádné záporné hodnoty.\n')

    return df


def gender_recoding(df):
    """
    Překódování kategorické proměnné 'Gender' na číselné hodnoty
    Mapování: "Male" -> 0, "Female" -> 1
    :param df: DataFrame
    :return:
        df - DataFrame s překódováným pohlavím
    """

    print('Probíhá překódování sloupce "Gender"...')
    if 'Gender' in df.columns:
        gender_mapping = {"Male":0, "Female": 1}
        df['Gender'] = df['Gender'].map(gender_mapping)
        print('Sloupec "Gender" byl převeden na číselné hodnoty.')
    else:
        print('Pozor! Sloupec "Gender" nebyl nalezen.')

    return df


def del_missing(df):
    """
    Odstraní řádky, kde v "Selector" chybí hodnota (NaN)
    :param df: DataFrame
    :return:
        df - DataFrame s odstranením chybících "Selector"
    """

    print('Probíhá kontrola v cílové proměnné...')
    raw_count = len(df)
    df.dropna(subset='Selector', inplace=True)
    new_count = len(df)
    deleted = raw_count - new_count
    if deleted > 0:
        print(f'Odstraněno {deleted} řádků kde byla chyba v cílové proměnné.')
    else:
        print('V cílové proměnné "Selector" nechyběly žádné hodnoty.')

    print(f'Aktuální počet řádků v datasetu: {new_count}')
    return df

def graf_shape(df):
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
def fix_age(df):
    """
    zobere dataframe a postará sa o to aby tam všetci boli mladší než 110 rokov
    :param df: DataFrame
    :return:
        dataframe z pozmenými hodnotami
    """
    df.loc[df['Age'] > 110, 'Age'] = np.nan
    return df



# --- Hlavní skript ---
if __name__ == "__main__":

    print('\nProbíhá načítání souboru...')
    # Načtení souboru
    path = 'liver-disease_data.csv'
    rdf, df = load_file(path)

    if df is not None:

        # Překódování cílové proměnné
        df = reformat_final(df)

        # Oprava fyziologicky nemožných hodnot (záporná data)
        print('\nKontrola a oprava záporných hodnot...')
        # Seznam sloupců kde zápor je nemožný
        columns_to_fix = ['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio']
        fix_age(df)

        # Iterativní oprava pro každý relevantní sloupec
        for column in columns_to_fix:
            if column in df.columns:
                data = negativ_num_correct(df, column)
            else:
                print(f'Sloupec {column} nebyl v datech nalezen.')
        print('\nKontrola záporných hodnot a oprava dokončena.')

        # Překódování pohlaví
        df = gender_recoding(df)

        # Odstranění řádků s chybějící cílovou hodnotou
        df = del_missing(df)

        print('\nPreprocessing dokončen.')
        print('Počet chybějících hodnot (NaN) v každém sloupci po základním zpracování:')
        print(df.isnull().sum())

        # Vizualizace rozdělení pohlaví
        print('Vytváření vizualizace pro rozdělení pohlaví...')
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x='Gender',
            discrete=True,
            hue=df['Gender'],
            shrink=.8).set_xticks([0, 1])
        plt.title('Rozdělení pacientů podle pohlaví')
        plt.xlabel('Pohlaví (0=Muž, 1=Žena)')
        plt.ylabel('Počet')
        plt.show()
        """ uloženie dát do novej tabulky pre okometrickú kontrolu"""
        #df.to_csv('liver-disease_data_edited.csv', index=False)
        """grafovanie"""
        #graf_shape(df)
        #get_corelation_matrix(df)
