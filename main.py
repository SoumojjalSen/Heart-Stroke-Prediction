import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, RocCurveDisplay

# Load the dataset
try:
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    # st.write(df.head())
except FileNotFoundError:
    st.error("Dataset not found. Please check the file path.")
    st.stop()

# Check if the target column exists
if 'stroke' not in df.columns:
    st.error("Target column 'stroke' not found in the dataset.")
    st.stop()

# Transform categorical columns
df.dropna(inplace=True)
df.drop(columns=['id'], inplace=True)
categorical_columns = df.select_dtypes(include='object').columns.tolist()
transformed_data = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Split the data into X and y
X = transformed_data.drop('stroke', axis=1)
y = transformed_data['stroke']

np.random.seed(42) # So that we can reproduce our results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg_grid = {"C": np.logspace(-4, 4, 20),
               "solver": ["liblinear"]}

smote = SMOTE()
X_res, y_res = smote.fit_resample(X_train, y_train)

gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

gs_log_reg.fit(X_res, y_res)

# User input fields
st.markdown("<h1 style='text-align: center;'>Heart Stroke Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This is a simple web app that predicts the likelihood of a person having a heart stroke based on their health data.</p>", unsafe_allow_html=True)

gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0, max_value=120, value=25)
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart Disease')
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])


    # Prepare user input for prediction
user_data = {
    'gender': gender,
    'age': age,
    'hypertension': int(hypertension),
    'heart_disease': int(heart_disease),
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}

user_df = pd.DataFrame(user_data, index=[0])
user_transformed = pd.get_dummies(user_df, columns=categorical_columns)
user_transformed = user_transformed.reindex(columns=X_res.columns, fill_value=0)
# st.write(user_transformed)

# Make prediction
prediction = gs_log_reg.predict(user_transformed)
# st.write(prediction)
prediction_proba = gs_log_reg.predict_proba(user_transformed)[0][1]
# st.write(gs_log_reg.predict_proba(user_transformed)[0])

st.write("")  # New line added before the existing one
st.markdown(
    "<div style='text-align: center;'>Prediction: You have a <b style='font-size:2em;'>{:.2f}%</b> chance of having a heart stroke.</div>".format(prediction_proba * 100),
    unsafe_allow_html=True
)