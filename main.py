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

# basic data analytics libs
import numpy as np
import pandas as pd
import seaborn as sns

# Principal component analysis
from sklearn.decomposition import PCA

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

# Geetting path to load data
path = os.getcwd()
# loading raw DataFrame
rdf = pd.read_csv(f"{path}/liver-disease_data.csv")

# Creating deep copy and replacing negative (non-sense) values
df = rdf.copy(deep=True)

# Assigning gender discrete values 
_ = {"Male": 0, "Female": 1}
df['Gender'] = df['Gender'].replace(_)
df[df < 0] = np.nan

"""
sns.histplot(
    data=df,
    x='Gender',
    discrete=True,
    hue=rdf['Gender'], 
    shrink = .8).set_xticks([0,1])
"""

# We can see missing entries
# df['Gender'].unique()