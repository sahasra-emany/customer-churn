import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv('cleanedchurn.csv')
sns.countplot(x='Churn', data= df)
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(), annot = True)
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']],hue='Churn',diag_kind='kde', height=2)
sns.violinplot(x='Contract', y='tenure', hue='Churn', data=df, split=True)

fig, axes = plt.subplots(1, 3, figsize=(12,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[0])
sns.boxplot(x='Churn', y='TotalCharges', data=df, ax=axes[1])
sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[2])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,3, figsize=(20,10))
sns.countplot(data=df, x='TechSupport', hue='Churn', stat='percent', ax=axes[0,0])
sns.countplot(data=df, x='StreamingTV',hue='Churn',  stat='percent',ax=axes[0,1])
sns.countplot(data=df, x='StreamingMovies',hue='Churn' ,stat='percent', ax=axes[1,1])
sns.countplot(data=df, x='OnlineSecurity', hue='Churn', stat='percent', ax= axes[0,2])
sns.countplot(data=df, x='OnlineBackup', hue='Churn', stat='percent' ,ax=axes[1,2])
sns.countplot(data=df, x='DeviceProtection', hue='Churn', stat='percent', ax=axes[1,0])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,3, figsize=(20,10))
sns.barplot(x='PhoneService', y='Churn', data=df, ax=axes[0,0])
sns.barplot(x='MultipleLines', y='Churn', data=df,ax=axes[0,1])
sns.barplot(x='InternetService', y='Churn', data=df, ax=axes[1,1])
sns.barplot(x='Contract', y='Churn', data=df, ax= axes[0,2])
sns.barplot(x='PaperlessBilling', y='Churn', data=df,ax=axes[1,2])
sns.barplot(x='PaymentMethod', y='Churn', data=df,ax=axes[1,0])
plt.tight_layout()
plt.show()

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies']
df['services_count'] = (df[service_cols] == 'Yes').sum(axis=1)
df['avg_monthly_cost'] = df['TotalCharges'] / df['tenure'].replace(0,1)