import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv('churn.csv')
df.shape
df.head()
df.columns
df.info()
df.isnull().sum()
f= df['TotalCharges'].isna()
df.loc[f,'TotalCharges']= df.loc[f,'MonthlyCharges']*df.loc[f,'tenure']
df.isnull().sum()
df = df.drop(columns=['gender'])
df = df.drop(columns=['SeniorCitizen'])
df = df.drop(columns=['Partner'])
df = df.drop(columns=['Dependents'])
df.columns
df.duplicated().sum()
df.duplicated(subset=df.columns.difference(['Churn'])).sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.nunique()
df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
(df['tenure'] > 0).all()
(df['MonthlyCharges'] >0 ).all()
(df['TotalCharges'] >= df['MonthlyCharges'] ).all()
df.nunique()
df['Churn']=df['Churn'].map({'No':0,'Yes':1})
df.dtypes
df.to_csv('cleanedchurn', index = False)
