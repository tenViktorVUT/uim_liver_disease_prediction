# %%%
# TODO: Model exportovať ako model a separe testing script
# - augmentace sa nepoužíva pri testovacích dátach
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


In order to succesfully run this script it requires
Data folder to be in the same directory as the script.
"""

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                       Imports
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# importing dependencies
# built-in libs
import os
import logging
from typing import List
import glob

import joblib
# Basic data analytics libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.base import BaseEstimator, TransformerMixin
# Principal component analysis
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    make_scorer,
    roc_curve
)

# Classificator XGBoost
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

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

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                       CORE FUNCS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def load_file(filename:str) -> pd.DataFrame:
    """
    Loads CSV data
    :param filename: Name of the file
    :return: Raw DF
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File {filename} was not found.')
        df = pd.read_csv(filename)
        logger.info(f'File {filename} loaded successfully. Shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Error loading file: {e}')
        return None
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performes cleaning and initial mapping
    Maps gender : Male->0, Female->1
    Maps Selector: 2->0 Healthy, 1->1 Disease
    Detects and removes impossibilities
    :param df: Raw DF
    :return: Cleaned DF
    """
    logger.info('Preprocessing data and cleaning physiological errors...')
    df = df.copy()
    # Target mapping
    #   Orig: 1 = Patient, 2 = Healthy
    #   New:  1 = Patient, 0 = Healthy
    if 'Selector' in df.columns:
        df['Selector'] = df['Selector'].map({1: 1, 2: 0})
    # Gender encoding
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    # Removing impossible negatives
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in df.columns if col not in ['Selector']]
    for col in feature_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logger.warning(f'Detected {neg_count} impossible negative values in "{col}". Converting to NaN.')
            df.loc[df[col] < 0, col] = np.nan
    # Age check
    if 'Age' in df.columns:
        df.loc[df['Age'] > 120, 'Age'] = np.nan # Oldest ever found 122
    return df
def del_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes entries with missing Selector
    :param (pd.DataFrame) df: DataFrame
    :return df: DataFrame with removed missing Selector entries
    """
    if 'Selector' not in df.columns:
        return df

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
def split_data(df: pd.DataFrame, seed: int=42):
    """
    Splits data into Train and Test sets (Test is LOCKED till final eval).
    :param df:
    :param seed:
    :return:
    """
    logger.info('Splitting data into Train and Test sets')

    X = df.drop('Selector', axis=1)
    y = df['Selector']

    # Stratify for imbalanced set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    logger.info(f'Train Shape: {X_train.shape}, Test Shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test

def get_pipeline(params: dict) -> ImbPipeline:
    """
    Constructing ML pipeline
    Ensures Imputation/Scaling happens inside the cross-valid folds
    :param params: Dictionary of XGBoost hyperparameters
    :return: Configured Pipeline object
    """
    skewed_cols = ['TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'A/G Ratio']

    pipeline = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            #("mrmr", mRMRSelector(k=5)),
            ('scaler', StandardScaler()),
            ('power', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('smote', SMOTETomek(random_state=42)),

            #('clf', RandomForestClassifier(**params, class_weight='balanced'))
            ('clf', XGBClassifier(**params, class_weight='balanced'))
        ])
    return pipeline


def train_model(X_train, y_train, best_params):
    """
    Trains the final Pipeline on the full training set using best params.
    Replaces "xgb_classify" func.
    :param X_train: train set data
    :param y_train: train set labels
    :param best_params: optimized hyperparameters
    :return: model named pipeline
    """
    logger.info('Training final model with best parameters...')

    #Set up for XGBooster
    final_params = best_params.copy()
    final_params.update({'n_jobs':-1, 'random_state': 42, 'eval_metric': 'logloss'})
    pipeline = get_pipeline(final_params)

    """
    # Set up for Random Forest
    final_params = best_params.copy()
    final_params.update({'n_jobs': -1, 'random_state': 42})
    pipeline = get_pipeline(
        final_params)
    """
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "model.pkl")

    return pipeline

def evaluate_model(
        model=None,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        mode: str = "cv",
        n_splits: int = 10,
        seed: int = 42
):
    """
    Evaluation function
    Modes:
        - "cv": Robust cross-validation with GridSearchCV + threshold tuning.
        - "test": Final evaluation on held-out test set with bootstrap CI + plots.

    :param model: Pretrained model (used in test mode).
    :param X, y: Training features/labels (used in cv mode).
    :param X_test, y_test: Test features/labels (used in test mode).
    :param mode: "cv" or "test"
    :param n_splits: Number of folds for CV (default 10).
    :param seed: Random seed.
    """
    if mode == "cv":
        logger.info(f"Starting {n_splits}-Fold Stratified CV evaluation...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_mccs, fold_accs, fold_thresholds = [], [], []
        feature_importances = pd.DataFrame(index=X.columns)

        param_grid = {
            'n_estimators': [100, 250, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        model_proto = XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"--- Fold {fold+1}/{n_splits} ---")
            train_x, val_x = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            grid = GridSearchCV(model_proto, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
            grid.fit(train_x, y_train)
            best_model = grid.best_estimator_

            y_proba = best_model.predict_proba(val_x)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 50)
            best_thresh, best_mcc = 0.7, -1
            for t in thresholds:
                y_pred_t = (y_proba > t).astype(int)
                mcc_t = matthews_corrcoef(y_val, y_pred_t)
                if mcc_t > best_mcc:
                    best_mcc, best_thresh = mcc_t, t

            y_pred_best = (y_proba > best_thresh).astype(int)
            filename = f"{"C:/Users/MSI/Documents/skola/UMI/uim_liver_disease_prediction"}/confusion_fold_{fold}.png"
            save_confusion_matrix(y_val, y_pred_best, filename)
            acc = accuracy_score(y_val, y_pred_best)

            fold_mccs.append(best_mcc)
            fold_accs.append(acc)
            fold_thresholds.append(best_thresh)

            try:
                fold_importances = pd.Series(best_model.feature_importances_, index=X.columns)
                feature_importances[f'fold_{fold+1}'] = fold_importances
            except Exception as e:
                logger.warning(f"Could not get feature importances in fold {fold+1}: {e}")

        logger.info("### CV Results ###")
        logger.info(f"Avg MCC: {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}")
        logger.info(f"Avg Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        logger.info(f"Avg Threshold: {np.mean(fold_thresholds):.4f}")

        plot_feature_importance(feature_importances)
        show_all_matrices(n_splits)
        cleanup_confusion_matrices()

    elif mode == "test":
        logger.info("Evaluating model on TEST set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        rng = np.random.RandomState(42)
        boot_scores = []

        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)

        for _ in range(1000):
            idx = rng.randint(0, len(y_test_arr), len(y_pred_arr))
            if len(np.unique(y_test_arr[idx])) < 2: continue
            boot_scores.append(matthews_corrcoef(y_test_arr[idx], y_pred_arr[idx]))
        ci_lower, ci_upper = np.percentile(boot_scores, [2.5, 97.5])

        print("\n" + "=-" * 30)
        print("   FINAL CLINICAL MODEL EVALUATION   ")
        print("=-" * 60)
        print(f"MCC: {mcc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"Kappa: {kappa:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=['Healthy','Patient']))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy','Patient'], yticklabels=['Healthy','Patient'])
        plt.title(f"Confusion Matrix (MCC={mcc:.2f})")
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", color='darkorange')
        plt.plot([0,1],[0,1],'--',color='navy')
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

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
        # Visualising gender distribution
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

        # debugging pring
        # plt.show()

        return None

    # Function calling
    logger.info('Visualising data...')
    get_corelation_matrix(df=df)
    plot_gender(df=df)
    graph_shape(df=df)

    plt.show()
    return None

def optuna_optimize(
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        metric: str = "mcc"
):
    """
    Flexible Optuna optimization func.
    Allows optimization for MCC, ROC AUC, or F1.
    Uses StratifiedKFold CV for stability.

    :param X: Training features
    :param y: Training labels
    :param n_trials: Number of Optuna trials
    :param metric: Metric to optimize ("mcc", "roc_auc", "f1")
    :return: best_params, best_value returns optimized prameters and best value
    """
    logger.info(f"Starting Optuna Optimization ({n_trials} trials) optimizing {metric.upper()}...")

    def objective(trial):
        # PARAMS FOR XGBoost
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'random_state': 42
        }

        pipeline = get_pipeline(params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        if metric == "mcc":
            scorer = make_scorer(matthews_corrcoef)
        elif metric == "roc_auc":
            scorer = "roc_auc"
        elif metric == "f1":
            scorer = make_scorer(f1_score)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            return scores.mean()
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best CV {metric.upper()}: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    return study.best_params, study.best_value


def plot_feature_importance(
    importances_df: pd.DataFrame
    ) -> None:
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
    
    return None

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix"
    ):
    """
    Plots a confusion matrix using pandas or numpy inputs.

    Parameters:
        y_true: array-like (pandas Series or numpy array)
        y_pred: array-like (pandas Series or numpy array)
        labels: list of labels (optional)
        title: graph title
    """

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')

    ax.set_title(title)
    plt.colorbar(im)

    # axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # write numbers inside boxes
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()
    
    return None

def save_confusion_matrix(
    y_true:pd.Series,
    y_pred:pd.Series,
    filename:str,
    labels:List[str] = None,
    title:str = None
    ):
    """
    saves a confusion matrix as png.
    same as plot confusion matrix but instead of plotting the matrix it saves it instead
    Parameters:
        y_true: array-like (pandas Series or numpy array)
        y_pred: array-like (pandas Series or numpy array)
        labels: list of labels (optional)
        title: graph title
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if title is None:
        title = filename
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    _, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest')

    ax.set_title(title)
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return None    

def show_all_matrices(n_splits=10) -> None:
    """
    displayes saved confusion matrices in 2 row format
    
    :param: n_splits: the number of saved matrices
    :return: None
    """
    cols = (n_splits // 2)
    rows = 2

    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 8))
    # Flattening for easier indexing
    axes = axes.flatten()  
    
    # Wrapping in list in case of 1 split
    if n_splits == 1:
        axes = [axes]

    for i in range(1, n_splits + 1):
        filename = f"confusion_fold_{i-1}.png"

        if not os.path.exists(filename):
            axes[i-1].set_title(f"Fold {i}\n(No Image)")
            axes[i-1].axis('off')
            continue

        img = mpimg.imread(filename)
        axes[i-1].imshow(img)
        axes[i-1].axis('off')
        axes[i-1].set_title(f"Fold {i}")

    plt.tight_layout()
    plt.show()
    
    return None

def cleanup_confusion_matrices() -> None:
    """
    goes through all the files in project folder and deletes saved confusion matrices
    """
    files = glob.glob("confusion_fold_*.png")

    if not files:
        logger.info("No confusion matrix images found to delete.")
        return

    for f in files:
        try:
            os.remove(f)
            logger.info(f"Deleted: {f}")
        except Exception as e:
            logger.info(f"Could not delete {f}: {e}")
            

    logger.info("Cleanup complete.")
    return None

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                       Main Script
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


if __name__ == "__main__":

    # Load
    filename = 'liver-disease_data.csv'
    df = load_file(filename)

    if df is not None:
        # Clean & Preprocess
        df = preprocess_data(df)
        df = del_missing(df)

        # Graphing (Optional, good for sanity check)
        graph_data(df)

        # Split
        #   The Pipeline does imputing and scaling
        X_train, X_test, y_train, y_test = split_data(df)
        # Check balance
        logger.info(f"Disease prevalence in Train: {y_train.mean():.2%}")

        # Optimizing with Optuna
        best_params, best_value = optuna_optimize(X_train, y_train, n_trials=50,metric='mcc')

        # 6. Train Final Model
        final_model = train_model(X_train, y_train, best_params)

        # CV evaluation
        evaluate_model(X=X_train, y=y_train, mode="cv", n_splits=10)
        # Final test evaluation with bootstrap CI
        evaluate_model(model=final_model, X_test=X_test, y_test=y_test, mode="test")