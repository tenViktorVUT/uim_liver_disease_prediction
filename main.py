# %%%

#TODO: SMOTE VS ADASIN
#TODO: confusion matrix plotting
#TODO: expand plotting
# TODO: MODEL A VÝBER HYPERPARAMETROV
# TODO: Funkcia na výber optimal param

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
from typing import Tuple

# NN
import tqdm
import shap

# Basic data analytics libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from qdm.pandas.tests.resample.test_resample_api import df_mult

# Principal component analysis
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
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
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from torch.utils.hipify.hipify_python import preprocessor

# Classificator XGBoost
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_file(filename: str) -> pd.DataFrame:
    """
    Loads CSV data under filename into 2 pandas DataFrames
    :param (str) filename: name of the file
    :return rdf: raw DataFrame
    :return df: deep copy DataFrame
    """

    try:
        # Finding path
        cwd = os.getcwd()
        path = os.path.join(cwd, filename)
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses data into a clean working DataFrame.
    Takes in raw data frame and replaces non-sense values with
    NaNs
    :param (pd.DataFrame) df: unprocessed DataFrame
    :return df: clean preprocessed data with nonsense values replaced by NaNs
    """

    ### REFORMATTING SELECTOR OUTPUT INTO BINARY VALUES
    if 'Selector' not in df.columns:
        # raises error in case of missing selector column
        raise NameError("Selector column missing in the DataFrame")

    # mapping binary values onto selector column
    # 1: pathological 2->0:healthy
    df['Selector'] = df['Selector'].map({1: 0, 2: 1})

    ### REPLACING NON-SENSE AGE
    df.loc[df['Age'] > 110, 'Age'] = np.nan

    ### REFORMATTING GENDER INTO BINARY, DISCRETE VALUES
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    ### CORRECTING NEGATIVE VALUES
    df[df < 0] = np.nan

    return df


def del_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes entries with missing Selector value  NaN
    :param (pd.DataFrame) df: DataFrame
    :return df: DataFrame with removed missing Selector entries
    """

    logger.info('Removing entries with missing Selector...')
    raw_count = len(df)

    # Dropping entries with missing selector
    df.dropna(subset='Selector', inplace=True)
    new_count = len(df)
    # Number of dropped entries
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
    :return None:
    """

    def graph_shape(df: pd.DataFrame) -> None:
        """
        Plots every feature from a DataFrame df.
        New figure is created after closing the previous one.
        :param df: DataFrame
        :return None:
        """
        logger.info('Creating histographs of features...')
        # Selects every numeric column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5))
        axes = axes.flatten()
        # Iterates over columns and for each one generates a histogram
        for i, col in enumerate(numeric_cols):
            sns.histplot(
                data=df, x=col,
                bins=30, hue='Selector',
                palette='rocket', ax=axes[i]
            )
            axes[i].set_title(col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(h_pad=8.0)
        #    plt.figure()
        #    sns.histplot(
        #        data=df, x=col,
        #        bins=30, hue='Selector',
        #        palette='rocket'
        #    )
        #    plt.title(col)
        #    plt.xlabel(col)
        #    plt.ylabel("Frequency")
        #    plt.show()

        return None

    def get_corelation_matrix(df: pd.DataFrame) -> None:
        """
        Generates correlation matrix and plots it in a heatmap.
        :param (pd.DataFrame) df: DataFrame
        :return None:
        """
        logger.info('Creating the correlation matrix...')
        corr = df.corr(numeric_only=True)

        # Plotting of the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis')
        plt.title("Correlation Matrix of Biomarkers")
        # plt.show()
        return None

    def plot_gender(df: pd.DataFrame) -> None:
        """
        Plots gender with hue showing Selector
        :param (pd.DataFrame) df: DataFrame
        :return: None
        """
        # Vizualizace rozdělení pohlaví
        logger.info('Creating the graf of gender division...')
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df,
            x='Gender',
            discrete=True,
            hue='Selector',
            palette='rocket',
            shrink=.8,
            multiple='stack'
        ).set_xticks([0, 1])

        plt.title('Rozdělení pacientů podle pohlaví')
        plt.xlabel('Pohlaví (0=Muž, 1=Žena)')
        plt.ylabel('Počet')
        # plt.show()
        return None

    # Function calling
    logger.info('Visualising data...')
    get_corelation_matrix(df=df)
    plot_gender(df=df)
    graph_shape(df=df)

    plt.show()
    return None


def fill_miss_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in missing values (NaNs) using KNNImputer for continuous data
    and most common value (modus) for discrete categoricacl data.
    :param (pd.DataFrame) df: DataFrame of preprocessed data
    :return df_imputed: DataFrame with filled-in NaNs
    """

    logger.info('Filling in missing values...')
    # Oddělení hodnoty kterou nechci upravovat
    selector_col = False
    if 'Selector' in df.columns:
        df_target = df['Selector'].copy(deep=True)
        df_features = df.drop('Selector', axis=1)
        selector_col = True
    else:
        df_features = df.copy()

    # Rozdělení sloupců na numerická a kategorické
    categorical_features = [
        col for col in df_features.columns
        if (df_features[col].dtype.name in ['object', 'category']) or (df_features[col].nunique(dropna=True) <= 5)
    ]
    numerical_features = [col for col in df_features.columns if col not in categorical_features]

    logger.info(
        f'Found {len(numerical_features)} numerical features.\n'
    )
    logger.info(
        f'Found {len(categorical_features)} categorical features.\n'
    )

    # Transformery
    numerical_transformer = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=5))
    ])  # 5 Sousedů
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
    new_columns_order = numerical_features + categorical_features
    df_imputed = pd.DataFrame(
        df_imputed_array,
        columns=new_columns_order,
        index=df_features.index
    )
    # Původní pořadí sloupců
    df_imputed = df_imputed[df_features.columns]

    # Připojení 'Selector'
    if df_target is not None:
        df_imputed = pd.concat([df_imputed, df_target], axis=1)

    logger.info('Finished filling in missing values')

    return df_imputed


def split_data(
        data: pd.DataFrame,
        seed: int = 42
) -> pd.DataFrame:
    """
    Splits data into training and validation data

    :param (pd.DataFrame) data: original data
    :param (int) seed: seed for random train/test split state
    :return train_x: training data
    :return val_x: validation data
    :return train_y: training for error
    :return val_y: validation error
    """

    logger.info('Splitting data into training and testing sets...')

    # Getting all columns names
    features = data.columns
    # Removing our Y from the list -> Selector
    features = [col for col in features if col != 'Selector']

    # Assigning X, Y
    Y = data.Selector
    X = data[features]
    (
        train_x, val_x, y_train, y_val
    ) = train_test_split(
        X, Y,
        test_size=0.25,
        random_state=seed
    )

    return train_x, val_x, y_train, y_val


def xgb_classify(
        train_x: pd.DataFrame, val_x: pd.DataFrame,
        y_train: pd.Series, y_val: pd.Series
) -> float:
    """
    Creates and trains XGB classifier
    :param (pd.DataFrame) train_x: training set
    :param (pd.DataFrame) val_x: validation set
    :param (pd.Series) train_y: training target
    :param (pd.Series) val_y: validation target
    :return Tuple:
    """

    logger.info('Training XGBClassifier...')

    # Creating model
    model = XGBClassifier(eta=0.005)
    model.fit(train_x, y_train)

    # Getting target predictions
    y_pred = model.predict(val_x)

    # Evaluating performance
    acc = accuracy_score(y_true=y_val, y_pred=y_pred)

    logger.info(f'Training complete. \nModel predicted target with overall accuracy: {acc}')
    return acc


def evaluate_model(X: pd.DataFrame, Y: pd.Series, n_splits: int = 10, seed: int = 42):
    """
    Does a robust evaluation of model with the help of stratificated cross validation.
    Implements a complete pipeline for training and evaluation:
        1. Splits data into 'n_splits' folds with the 'StratifiedKFold'
        2. For each fold:
            a. Computes 'scale_pos_weight' for class balancing in XGBoost
            b. Launches 'GridSearchCV' (3-fold embedded CV)  to find the best
                hyperparameters in training data. Optimizes on 'roc_auc'.
            c. On validation data does a manual search for optimal threshold
                for the maximum MCC
            d. Saves final MCC and ACC for this fold
            e. Saves score from the training data for overfitting diagnostics
        3. Prints average MCC, ACC, optimal threshold and training score
        4. Shows graph of the average importance across folds

    :param X: pd.DataFrame
                DataFrame with final features
    :param Y: pd.Series
                Series with target value
    :param n_splits: int, optional
                Number of folds for cross validation (default: 10)
    :param seed: int, optional
                Random state to find out reproductability (default: 42).
    :return:
        None
    """
    logger.info(f'Starting the validation of the model ({n_splits}-Fold Stratified K-Fold)...')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Seznamy pro ukládání výsledků z foldů
    fold_accuracies = []
    fold_mccs = []
    fold_thresholds = []
    # Seznamy pro diagnostiku přeučení
    fold_train_mccs = []
    fold_train_aucs = []
    # DataFrame pro ukládání důležitosti rysů z foldů
    feature_importances = pd.DataFrame(index=X.columns)

    # =========== GRID a PIPELINE pro scale_pos_weight ===========
    param_grid = {
        'n_estimators': [100, 250, 400],  # Počet stromů
        'max_depth': [3, 5, 7, 9],  # Hloubka stromů
        'learning_rate': [0.01, 0.05, 0.1],  # Kroky učení
        'subsample': [0.7, 0.9, 1.0],  # Procento řádků (pacientů)
        'colsample_bytree': [0.7, 0.9, 1.0]  # Procento rysů (sloupců)
    }
    # Výpočet váhy
    try:
        scale_pos_weight = (Y == 0).sum() / (Y == 1).sum()
        logger.info(f'Calculated scale_pos_weight for imbalance: {scale_pos_weight:.4f}')
    except ZeroDivisionError:
        logger.warning('Positive class (1) has zero samples. Setting scale_pos_weight to 1.')
        scale_pos_weight = 1
    # Prorotyp pro XGBoost
    model_proto = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    # =========== GRID a PIPELINE pro SMOTE ===========
    # Pre budúcnosť radšej ponechaný
    """
    param_grid = {
        'model__n_estimators': [100, 250],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0]
    }

    param_grid = {
        'model__n_estimators': [100, 250, 400],     # Počet stromů
        'model__max_depth':    [3, 5, 7, 9],      # Max hloubka stromu
        'model__learning_rate':[0.01, 0.05, 0.1],    # Rychlost učení
        'model__subsample':    [0.7, 0.9, 1.0]      # % dat pro trénování každého stromu
        'model__colsample_bytree': [0.7, 0.9, 1.0]
    }

    # Pipeline pro aplikaci SMOTE
        # SMOTE jen na trénovací data !
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=seed)),
        ('model', XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])
    """

    # Hlavní smyčka pro křížovou validaci
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        logger.info(f'--- Fold {fold + 1}/{n_splits} ---')
        # Rozdělení dat na trénovací a validační pro tento fold
        train_x, val_x = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]

        # GridSearchCV
        # Hledání nejlepších parametrů, ale pouze na trénovacích datech
        # 'cv=3' -> 3-Fold CV uvnitř jednoho foldu
        # Optimalizace na "roc_auc"
        grid_search = GridSearchCV(
            estimator=model_proto,  # pipeline=SMOTE, modelproto=scale_pos_weight
            param_grid=param_grid,
            scoring='roc_auc',  # Ješte může být f1_weighted nebo mcc
            cv=3,  # 3-fold CV v tomto foldu
            n_jobs=-1,
            verbose=0  # Pro více logů nastavit na 1
        )

        logger.info(f'Fold {fold + 1}: Starting GridSearchCV with SMOTE...')
        grid_search.fit(train_x, y_train)
        # Nejlepší model z tohoto foldu
        # best_pipeline = grid_search.best_estimator_        # pipeline -> SMOTE
        best_model = grid_search.best_estimator_  # model -> weight

        logger.info(f'Fold {fold + 1}: Best params found: {grid_search.best_params_}')
        logger.info(f'Fold {fold + 1}: Best internal CV score (ROC AUC): {grid_search.best_score_:.4f}')

        # Hledání optimálního prahu (Thresholding) (na validačních)

        # y_pred_proba = best_pipeline.predict_proba(val_x)[:,1]     # -> SMOTE
        # Pravděpodobnost pro třídu 1 (nemocný)
        y_pred_proba = best_model.predict_proba(val_x)[:, 1]  # -> weight
        # 50 prahů
        thresholds = np.linspace(0.1, 0.9, 50)  # 50 prahů
        best_mcc = -1.0  # -1 -> 1 range
        best_thresh = 0.5

        for thresh in thresholds:
            # Pravděpodobnosti -> binární predikce
            y_pred_thresh = (y_pred_proba > thresh).astype(int)
            # MCC pro tento práh
            current_mc = matthews_corrcoef(y_val, y_pred_thresh)

            if current_mc > best_mcc:
                best_mcc = current_mc
                best_thresh = thresh
        logger.info(f'Fold {fold + 1}: Best threshold found {best_thresh:.4f} (gives MCC: {best_mcc:.4f}')

        # Finální predikce s nejlepším prahem
        y_pred_best = (y_pred_proba > best_thresh).astype(int)

        # Predikce na validačních datech
        # y_pred = best_pipeline.predict(val_x)

        # Evaluace (na VALIDAČNÍCH), s optimálním prahem
        acc = accuracy_score(y_true=y_val, y_pred=y_pred_best)
        mcc = best_mcc

        fold_accuracies.append(acc)
        fold_mccs.append(mcc)
        fold_thresholds.append(best_thresh)

        # Diagnostika přeučení (na trénikových datech)
        # y_train_proba = best_pipeline.predict_proba(train_x)[:,1]      # -> SMOTE
        y_train_proba = best_model.predict_proba(train_x)[:, 1]  # -> weight
        y_train_pred = (y_train_proba > best_thresh).astype(int)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)

        fold_train_mccs.append(train_mcc)
        fold_train_aucs.append(train_auc)

        logger.info(f'Fold {fold + 1} Train MCC: {train_mcc:.4f} | Val MCC: {mcc:.4f}')
        logger.info(f'Fold {fold + 1} Train AUC: {train_auc:.4f} | Val AUC: {grid_search.best_score_:.4f}')

        # Uložení důležitých rysů
        try:
            # model_in_pipeline = best_pipeline.named_steps['model']     # -> SMOTE
            # fold_importances = pd.Series(model_in_pipeline.feature_importances_, index=X.columns)   # -> SMOTE
            fold_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            feature_importances[f'fold_{fold + 1}'] = fold_importances
        except Exception as e:
            logger.warning(f'Could not get feature importances in fold {fold + 1}. Error: {e}')

    # Finální výsledky
    logger.info(f'Validation done.')
    logger.info('### Average validation scores ###')
    logger.info(f'Average accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}')
    logger.info(f'Average MCC: {np.mean(fold_mccs):.4f} +/- {np.std(fold_mccs):.4f}')
    logger.info(f'Average optimal threshold: {np.mean(fold_thresholds):.4f} +/- {np.std(fold_thresholds):.4f}')

    logger.info(' ### Average Training Scores (Overfitting Check) ###')
    logger.info(f'Average Train MCC: {np.mean(fold_train_mccs):.4f} +/- {np.std(fold_train_mccs):.4f}')
    logger.info(f'Average Train AUC: {np.mean(fold_train_aucs):.4f} +/- {np.std(fold_train_aucs):.4f}')
    # Plot výsledek
    plot_feature_importance(feature_importances)


def plot_feature_importance(importances_df: pd.DataFrame):
    """
    Plots the average importance of features across all the folds.
    :param importances_df: pd.DataFrame
            DataFrame, indexes are feature names and columns represent
            importance from individual folds
    :return:
        None - shows a matplotlib graph
    """
    logger.info('Plotting the average importance of features...')
    # Počítaní průměru a  Std
    mean_importance = importances_df.mean(axis=1)
    std_importance = importances_df.std(axis=1)

    plot_df = pd.DataFrame({
        'mean_importance': mean_importance,
        'std_importance': std_importance
    })
    # Seřazení sestupně od nejdůležitějšího
    plot_df = plot_df.sort_values(by='mean_importance', ascending=False)

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(plot_df))
    means = plot_df['mean_importance'].values
    errs = plot_df['std_importance'].values
    # Vykreslení horizont bar plotu
    plt.barh(y_pos, means, xerr=errs, align='center', color='tab:blue', ecolor='gray')
    plt.yticks(y_pos, plot_df.index)
    plt.gca().invert_yaxis()  # Nejlepší feature nahoře
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, clinically relevant features from the existing data.
    Has to be called after 'fill_miss_values', to avoid /NaN.
    Calculates 2 new features
        1. 'De_Ritis_Ratio': AST/ALT ratio
        2. 'AG_Ratio_Calculated': Albumin/Globulin ratio
    :param df: pd.DataFrame
                DataFrame with already filled values
    :return:
        pd.DataFrame
            Copy of original DataFrame with added features
    """
    logger.info('Creating new features...')
    df_out = df.copy()
    # Konstanta pro nedělení 0
    epsilon = 1e-6

    # De Ritisův poměr (AST/ALT)
    # Sgot = AST, Sgpt = ALT
    if 'Sgot' in df_out.columns and 'Sgpt' in df_out.columns:
        sgpt_safe = df_out['Sgpt'].replace(0, epsilon)
        df_out['De_Ritis_Ratio'] = df_out['Sgot'] / (sgpt_safe + epsilon)
        logger.info('Created a feature "De_Ritis_Ratio" (Sgot/Sgpt).')
    else:
        logger.warning('Columns "Sgot" or "Sgpt" are missing for the calculation of "De_Ritis_Ratio"')

    if 'TP' in df_out.columns and 'ALB' in df_out.columns:
        globulin = df_out['TP'] - df_out['ALB']
        globulin_safe = globulin.replace(0, epsilon)
        df_out['AG_Ratio_Calculated'] = df_out['ALB'] / (globulin_safe + epsilon)
        logger.info('Created a new feature "AG_Ratio_Calculated".')
    else:
        logger.warning('Column "TP" or "ALB" is missing and cannot calculate A/G Ratio.')

    # Odstranění původního kdyby byl neposlehlivý

    return df_out


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Application of RobustScales on all numerical features apart from "Gender" and
    "Selector". "Selector" has to be in the DataFrame.
    RobustScales helps with the outliers.
    :param df: pd.DataFrame
                DataFrame that is supposed to scale. Needs a "Selector".
                Should be called after log transformation.
    :return:
        pd.DataFrame
            DataFrame with scaled features.
    """
    logger.info('Application of RobusScaler on the data...')
    df_scaled = df.copy()

    # Oddělení cílové proměnné
    target = None
    if 'Selector' in df_scaled.columns:
        target = df_scaled.pop('Selector')
    else:
        logger.error('Error with scaling: "Selector" is missing in the data.')
        return df
    # Oddělení Gender
    gender = None
    if 'Gender' in df_scaled.columns:
        gender = df_scaled.pop('Gender')  # Neškáluji Gender

    features_to_scale = df_scaled.columns

    if not features_to_scale.empty:
        scaler = RobustScaler()
        df_scaled_array = scaler.fit_transform(df_scaled)
        # Dataframe zpátky
        df_scaled = pd.DataFrame(df_scaled_array, columns=features_to_scale, index=df_scaled.index)

    # Vrácení Selector a Gender
    if gender is not None:
        df_scaled = pd.concat([df_scaled, gender], axis=1)
    if target is not None:
        df_scaled = pd.concat([df_scaled, target], axis=1)

    logger.info('Scaling was successful.')
    return df_scaled


def clip_physiological_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clipping extreme values (outliers) based on quantiles.
    Helps the model not to learn from extreme values.
    Clipping at 99.5 percentile.

    This func was REPLACED by 'apply_log_transform', which appeared more robust.
    :param df: DataFrame
                DataFrame, with values for clipping
    :return:
        df_clipped: DataFrame with clipped values
    """
    logger.info('Clipping extreme outliers based on 99.5th percentile...')
    df_clipped = df.copy()

    features_to_clip = [
        'TB',
        'DB',
        'Alkphos',
        'Sgpt',
        'Sgot',
        'TP',
        'ALB',
        'A/G Ratio'
    ]

    for col in features_to_clip:
        if col in df_clipped.columns:
            # Výpočet percentilu
            # Ošetření NaN
            upper_limit = df_clipped[col].dropna().quantile(0.995)
            values_to_be_clipped_count = (df_clipped[col] > upper_limit).sum()

            if values_to_be_clipped_count > 0:
                logger.info(
                    f'Clipping {values_to_be_clipped_count} values in "{col}" (setting max to {upper_limit:.2f}')
                # Clipping
                df_clipped[col] = df_clipped[col].clip(upper=upper_limit)
            else:
                logger.info(f'No values to clip in "{col}".')
        else:
            logger.warning(f'Column "{col}" not found for clipping.')
    logger.info('Clipping complete.')
    return df_clipped


#FIXME: Logtransform vie pokaziť dáta
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses logarithmical transform (log1p) on strongly skewed features.
    Call AFTER fill_miss_values and BEFORE scaling.
    :param df:
        DataFrame with filled values.
    :return:
        pd.DataFrame
            Copy of DataFrame with transformed columns
    """
    logger.info('Applying Log Transform (log1p) to skewed features...')
    df_transformed = df.copy()

    # Rysy s long-tail distribution
    features_to_transform = [
        'TB',
        'DB',
        'Alkphos',
        'Sgpt',
        'Sgot'
    ]

    for col in features_to_transform:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col])
            logger.info(f'Log transform applied to "{col}".')
        else:
            logger.warning(f'Column "{col}" not found for log transform.')
    logger.info('Log transform complete.')
    return df_transformed



# %%%
#
#
#
# --- Hlavní skript ---
if __name__ == "__main__":

    logger.info('Loading data file...')
    # Načtení souboru
    path = 'liver-disease_data.csv'
    rdf, df = load_file(path)

    #%%%
    if df is not None:
        # Základní předzpracování
        df = preprocess_data(df=df)
        # display(df)
        # Odstranění řádků s chybějící cílovou hodnotou
        df = del_missing(df=df)
        #   Ořezání extrémních hodnot -> menší MCC než logaritmická
        #   df = clip_physiological_values(df=df)
        display(df)
        
        #%%%
        # Doplnění chybějících hodnot
        df = fill_miss_values(df=df)
        # Vytvoření nových features
        df = create_features(df=df)
        display(df)
        # Aplikace logaritmické transformace
        # df = apply_log_transform(df=df)
        # display(df)
        # Škálování
        # df = scale_data(df=df)
        # display(df)
        logger.info('Data preprocessing completed')
        # print('Počet chybějících hodnot (NaN) v každém sloupci po základním zpracování:')
        # print(df.isnull().sum()) # Správně nuly...

        #%%%
        graph_data(df=df)

        #%%%
        # train_x, val_x, y_train, y_val = split_data(data=df)
        # acc = xgb_classify(train_x, val_x, y_train, y_val)
        if 'Selector' not in df.columns:
            logger.error('"Selector" is not in the DataFrame, cannot continue.')
        else:
            Y = df['Selector'].copy()
            X = df.drop('Selector', axis=1).copy()

            evaluate_model(X, Y)
            
        #TODO: confusion matrix
        confusion_matrix()
#TODO: feature importance pre and post log transform