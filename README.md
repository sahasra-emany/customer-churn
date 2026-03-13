# Customer Churn Prediction App

## Overview
This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to discontinue a service so that businesses can take proactive retention measures.
The project includes exploratory data analysis, feature engineering, model training and evaluation, and deployment of an interactive web application.

---
## Project Highlights
- Performed detailed Exploratory Data Analysis (EDA) to understand churn patterns
- Built a complete machine learning pipeline using **Scikit-learn**
- Compared multiple models including **Logistic Regression, Random Forest, and XGBoost**
- Interpreted model behavior using **odds ratios**
- Deployed an interactive **Streamlit web application** for real-time prediction
---
## Feature Engineering
Several features were engineered to improve model performance:
- OneHotEncoding for categorical variables
- Standard scaling for numerical variables
- Service Count (number of active services)
- Average Charges derived from billing data
- Handling class imbalance using class_weight = 'balanced'
---
## Models Used

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Logistic Regression | 0.73 | 0.49 | 0.82 | 0.62 | 0.84 |
| Random Forest | 0.77 | 0.60 | 0.44 | 0.51 | 0.81 |
| XGBoost | 0.75 | 0.52 | 0.65 | 0.58 | 0.81 |

Logistic Regression achieved the best **ROC-AUC score** and higher recall.

---
## Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross Validation and Bias-Variance Trade off
## Key Insights
- Customers with **month-to-month contracts** are more likely to churn.
- Higher **monthly charges** correlate with increased churn probability.
- Customers with **fewer active services** show higher churn rates.
- **Longer tenure customers** are significantly less likely to churn.
---
## Deployment
The trained machine learning pipeline was deployed using **Streamlit**, allowing users to:
- Enter customer attributes
- Predict churn probability usin
- Interact with model inputs through a web interface
---
## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
---
## Future Prospects
- Implement customer segmentation analysis
- Integrate API endpoints for production use
- Add Docker containerization for scalable deployment
---
## Author
Sahasra Emany
