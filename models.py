import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
df = pd.read_csv('cleanedchurn.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
numeric_cols = X.select_dtypes(include=['int64','float64']).columns
categorical_cols = X.select_dtypes(include=['object','category']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)])
models = {
    "lr": LogisticRegression(class_weight='balanced', max_iter=1000),
    "rf": RandomForestClassifier(class_weight='balanced', random_state=42),
    "xgb": XGBClassifier(scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1]),
                            eval_metric='logloss',random_state=42)}

results = {}
plt.figure(figsize=(8,6))
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)}
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
results_df = pd.DataFrame(results).T
print(results_df)
results = {}
plt.figure(figsize=(8,6))
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)}
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
results_df = pd.DataFrame(results).T
print(results_df)
lr = Pipeline(steps=[('preprocessing', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))])
lr.fit(X_train, y_train)
rf = Pipeline(steps=[('preprocessing', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))])
rf.fit(X_train, y_train)
xgb = Pipeline(steps=[('preprocessing', preprocessor),
    ('model', XGBClassifier(scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1]),
                             eval_metric='logloss',random_state=42))])
xgb.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(cm_lr).plot(ax=axes[0])
axes[0].set_title("Logistic Regression")
ConfusionMatrixDisplay(cm_rf).plot(ax=axes[1])
axes[1].set_title("Random Forest")
ConfusionMatrixDisplay(cm_xgb).plot(ax=axes[2])
axes[2].set_title("XGBoost")
plt.tight_layout()
plt.show()
print("Train score:", lr.score(X_train, y_train))
print("Test score:", lr.score(X_test, y_test))
print("Train score:", rf.score(X_train, y_train))
print("Test score:", rf.score(X_test, y_test))
print("Train score:", xgb.score(X_train, y_train))
print("Test score:", xgb.score(X_test, y_test))
cv_scores = cross_val_score(lr, X, y, cv=5)
print("CV Mean:", cv_scores.mean())
cv_scores = cross_val_score(rf, X, y, cv=5)
print("CV Mean:", cv_scores.mean())
cv_scores = cross_val_score(xgb, X, y, cv=5)
print("CV Mean:", cv_scores.mean())
feature_names = lr.named_steps['preprocessing'].get_feature_names_out()
coefs = lr.named_steps['model'].coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs})
print(coef_df)
df['coefficients'] = coef_df["Coefficient"]
df.to_csv("churn_predictions.csv", index=False)
import joblib
joblib.dump(pipe, "churn_model.pkl")
joblib.dump(lr, 'model.pkl')