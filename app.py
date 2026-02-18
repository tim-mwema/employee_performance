import streamlit as st
import pandas as  pd
import numpy as  np 
import joblib

#loading the model
model=joblib.load('model.pkl')

st.title("Employee Performance Prediction")
st.write("Enter the values scores of the following areas for the employee to predict performance")
EmplEmpEnvironmentSatisfaction=st.number_input("Satisfaction level(1=low,2=medium,3=high,4=very high)",min_value=1.0,max_value=4.0,step=1.0)
EmpLastSalaryHikePercent=st.number_input("Percent salary added",min_value=1.0,max_value=30.0,step=1.0)
EmpWorkLifeBalance=st.number_input("Worklife_Balance score(1=low,2=medium,3=high,4=very high)",min_value=1.0,max_value=4.0,step=1.0)

if st.button("Predict"):
    input_features=np.array([[EmplEmpEnvironmentSatisfaction,EmpLastSalaryHikePercent,EmpWorkLifeBalance]])
    prediction=model.predict(input_features)#For my model to predict
    PerformanceRating={1:"Low",2:"Good",3:"Excellent",4:"Outstanding"}#doing my output mapping
    Predicted_Performance=PerformanceRating[int(prediction[0])]
    #to display my model output
    st.success(f"The predicted performance is:{Predicted_Performance}")