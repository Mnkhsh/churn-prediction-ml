This project builds an end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. It includes data preprocessing, feature engineering, model training, hyperparameter tuning and model deployment using a saved .pkl file.

Project Highlights

Built a complete ML pipeline using Python, scikit-learn, and pandas

Performed data cleaning, encoding, scaling, and class balancing

Tuned a Random Forest classifier for optimal recall

Achieved:

Recall: ~0.76

ROC-AUC: ~0.83

F1 Score: ~0.62

Explained the model using SHAP (global + local explanations)

Saved the final production model to final_churn_model.pkl

🧠 Modeling Approach
1. Data Preprocessing

Missing values handled

TotalCharges converted to numeric

One-hot encoding for categorical variables

Standardization for numeric features

Stratified train/test split

2. Model Training

Random Forest with tuned parameters:

n_estimators = 300
max_depth = 10
min_samples_split = 10
min_samples_leaf = 4
bootstrap = True
class_weight = "balanced"

3. Evaluation Metrics
Metric	Score
Accuracy	~0.75
Precision	~0.53
Recall	~0.76
F1	~0.62
ROC-AUC	~0.83
4. Feature Importance

Top predictors:

Contract: Month-to-month

Tenure

Total Charges

Monthly Charges

Online Security

Tech Support

Fiber Optic Internet

5. Explainability (SHAP)

Global summary plot

Local force plot for individual predictions

Transparent insights into churn-driving factors

💾 Model Saving & Loading
import joblib
model = joblib.load("final_churn_model.pkl")
model.predict(X_test.head(1))

🛠 Technologies Used

Python

pandas

numpy

scikit-learn

SHAP

matplotlib

📈 Future Improvements

Streamlit web app for user input

Compare with gradient boosting (XGBoost, LightGBM)

Hyperparameter tuning with Bayesian Optimization

Deploy via FastAPI or AWS Lambda

🙌 Author

Project created by Mariam Nikath
Machine Learning & AI Engineer
