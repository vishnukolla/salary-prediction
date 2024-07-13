import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# Load data
data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\salary.csv")

# Streamlit app title
st.title("Predict Your Salary Application")
st.write("This is an application for knowing how much will be your salary based on your Grades!")

# Checkbox to view data
check_data = st.checkbox("View the data")
if check_data:
    st.write(data.head(3))

st.write("Now let's find how much the salary will be with other parameters")

# Fill missing values
data['experience'].fillna(0, inplace=True)
data['test_score'].fillna(data['test_score'].mean(), inplace=True)

# Features and target variable
X = data.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = data.iloc[:, -1]
# Sliders for user input
experience = st.slider("How much is your experience?", int(X.experience.min()), int(X.experience.max()), int(X.experience.mean()))
test_score = st.slider("How much is your test score?", int(data.test_score.min()), int(data.test_score.max()), int(data.test_score.mean()))
interview_score = st.slider("How much is your interview score?", int(data.interview_score.min()), int(data.interview_score.max()), int(data.interview_score.mean()))

# Linear regression model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict([[experience, test_score, interview_score]])[0]

# Predict button
if st.button("PREDICT THE SALARY"):
    st.header(f"Your Salary Prediction is Rupees {int(predictions)}")
