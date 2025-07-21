# Employee Salary Prediction Web App


## 📖 Overview

This project is a machine learning-powered web application that predicts whether an individual's annual income is likely to be more or less than $50,000.

The application features a user-friendly interface built with Streamlit, where users can input an individual's details and receive an instant prediction from the best-performing machine learning model.

---

## ✨ Features

-   **Interactive UI:** A clean and simple web interface for easy user input.
-   **Multiple Model Training:** The application trains five different classification models:
    -   Logistic Regression
    -   Random Forest
    -   K-Nearest Neighbors (KNN)
    -   Support Vector Machine (SVM)
    -   Gradient Boosting
-   **Automated Model Selection:** The app automatically evaluates all trained models and uses the one with the highest accuracy for predictions.
-   **Advanced Data Preprocessing:** Implements a specific, detailed data cleaning and feature engineering workflow before model training.
-   **Real-time Predictions:** Instantly predicts the income bracket based on user-provided data.

---

## ⚙️ How It Works

The application follows a standard machine learning pipeline:

1.  **Data Loading:** The application starts by loading the `adult.csv` dataset.
2.  **Data Preprocessing:** It performs several cleaning and transformation steps:
    -   Handles missing values by replacing them with a new `'Others'` category.
    -   Filters the dataset based on age, education, and workclass to remove outliers and irrelevant data.
    -   Converts all categorical text features (like `workclass`, `occupation`, `gender`) into numerical format using `LabelEncoder`.
3.  **Model Training:** The preprocessed data is split into training and testing sets. The five different machine learning models are then trained on this data.
4.  **Model Evaluation & Selection:** Each model's performance is measured by its accuracy on the test set. The model with the highest accuracy is selected as the "best model."
5.  **Prediction:** When a user enters data into the Streamlit interface, the application applies the same preprocessing steps to the input and feeds it to the best model to generate a prediction.

---

## 🛠️ Technologies Used

-   **Backend & ML:** Python, Scikit-learn, Pandas, NumPy
-   **Frontend:** Streamlit

---

## 🚀 Setup and Installation

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**

```bash
https://github.com/SHREYASH-KUMAR1601/Employee_Salary_Prediction.git
cd employee-salary-prediction
