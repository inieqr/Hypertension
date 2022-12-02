# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 01:57:14 2022

@author: Anon
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for prediction
def hypertension_detection(input_data):
    
  
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    
    if(prediction[0] == 0):
      print("Low blood preassure")
    else:
      print("High blood preassure")
      
    
  
def main():
    
    
    # giving a title
    st.title('Ontology-based Approach in Medical Health using Hypertension as a case study')
    
    st.header('Enter the following information')
    
    # getting the input data from the user
    
    gender = st.text_input('Gender:     0 - Female, 1 - Male, 2 - Others')
    age = st.text_input('Age')
    heart_disease = st.text_input('Do you have heart disease?       1 - Yes, 0 - No ')
    ever_married = st.text_input('Ever married?     1 - Yes, 0 - No')
    work_type = st.text_input('Work Type:       0 - Government, 1 - Never worked, 2 - Private, 3 - Self-employed, 4 - Children')
    Residence_type = st.text_input('Residence Type:     0 - Rural, 1 - Urban')
    avg_glucose_level = st.text_input('Average glucose level in blood')
    bmi = st.text_input('Body Mass Index (BMI)')
    smoking_status = st.text_input('Smoking status:     0 - Unknown, 1 - Formerly smoked, 2 - Never smoked, 3 - Smokes')
    stroke = st.text_input('Ever had a stroke?      1 - Yes, 0 - No')
    
    
    # code for prediction
    testing = ''
    
    # creating a button for Prediction
    
    if st.button('Detection Result'):
        testing = hypertension_detection([gender, age, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke])


    st.success(testing)
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
