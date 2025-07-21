# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# MODEL TRAINING FUNCTION
@st.cache_data
def train_model():
    try:
        df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\adult 3.csv")
    except FileNotFoundError:
        st.error("Error: File not found. Please ensure the dataset file is in the correct directory.")
        return None, None, None, None

    df.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df.rename(columns={'sex': 'gender'}, inplace=True)

    for col in df.select_dtypes(['object']):
        df[col] = df[col].str.strip()

    # 2. Replace '?' with 'Others'
    df.workclass.replace({'?': 'Others'}, inplace=True)
    df.occupation.replace({'?': 'Others'}, inplace=True)
    df['native-country'].replace({'?': 'Others'}, inplace=True)
    df['native-country'] = np.where(df['native-country'] != 'United-States', 'Others', 'United-States')

    # 3. Filter rows based on specific criteria
    df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
    df = df[(df['age'] >= 17) & (df['age'] <= 75)]
    df = df[(df['education-num'] >= 5) & (df['education-num'] <= 16)]
    
    # 4. Drop unnecessary columns
    df.drop(['fnlwgt', 'education'], axis=1, inplace=True)

    # 5. Label Encode all categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('income')

    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        feature_encoders[col] = le
        
    # Encode target variable separately
    target_encoder = LabelEncoder()
    df['income'] = target_encoder.fit_transform(df['income'])

    # --- MODEL TRAINING ---
    X = df.drop('income', axis=1)
    y = df['income']
    
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    best_model_name = ""
    best_accuracy = 0.0
    best_model_pipeline = None

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_pipeline = pipe

    st.write(f"The best model is **{best_model_name}** with an accuracy of **{best_accuracy:.2f}**.")
    
    # Return the feature names along with other objects
    return best_model_pipeline, feature_encoders, target_encoder, feature_names


# STREAMLIT APP INTERFACE
st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("Employee Salary Prediction")
st.write("""
Enter the employee's details in the sidebar to get a salary prediction.
""")

# --- Load Model and Data ---
with st.spinner('Training models...'):
    best_model, feature_encoders, target_encoder, feature_names = train_model()

if best_model is None:
    st.stop()

st.sidebar.header("Employee Details")

def user_input_features(feature_names, feature_encoders):
    inputs = {}
    
    inputs['age'] = st.sidebar.slider("Age", 17, 75, 35)
    
    # Create dropdowns for categorical features using the classes from the fitted encoders
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        if col in feature_encoders:
            encoder = feature_encoders[col]
            label = col.replace('-', ' ').title()
            options = list(encoder.classes_)
            selected_option = st.sidebar.selectbox(label, options)
            inputs[col] = encoder.transform([selected_option])[0]

    # Map for education-num, which was not label encoded
    education_map = {
        5: '9th', 6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college', 
        11: 'Assoc-voc', 12: 'Assoc-acdm', 13: 'Bachelors', 14: 'Masters', 
        15: 'Prof-school', 16: 'Doctorate'
    }
    education_label = st.sidebar.selectbox("Education Level", list(education_map.values()), index=8)
    inputs['education-num'] = [k for k, v in education_map.items() if v == education_label][0]

    inputs['capital-gain'] = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
    inputs['capital-loss'] = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
    inputs['hours-per-week'] = st.sidebar.slider("Hours per Week", 1, 99, 40)
    
    # Create a DataFrame from inputs
    features = pd.DataFrame(inputs, index=[0])

    return features[feature_names]

# Get user input
input_df = user_input_features(feature_names, feature_encoders)

# Display the user input in the main area
st.subheader("Your Input (Encoded):")
st.table(input_df)

if st.button("Predict Salary"):
    prediction_encoded = best_model.predict(input_df)    
    prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader("Prediction Result")
    if prediction_label == '>50K':
        st.success(f"The predicted income is **{prediction_label}**")
    else:
        st.info(f"The predicted income is **{prediction_label}**")

