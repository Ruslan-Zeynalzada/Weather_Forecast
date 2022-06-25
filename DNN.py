import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import tensorflow as tf
import pickle

header = st.container()
description = st.container()
data = st.container()
X_test = st.container()
modeling = st.container()
btn_desc = st.sidebar.button("Description")


with header: 
    st.title("The program is to predict if it will rain tomorrow or not")
    st.markdown("* First you have to enter **inputs** on the left of the screen")
    st.markdown("* Then you can press **Predict** button and see the results")
    st.markdown("* If you want to see the variables's description press **Description** button")
if btn_desc : 
    with description :
        st.header("Variables' Description")
        st.markdown("* **MinTemp** - The minimum temperature in degrees celsius")
        st.markdown("* **MaxTemp** - The maximum temperature in degrees celsius")
        st.markdown("* **Rainfall** - The amount of rainfall recorded for the day in mm")
        st.markdown("* **WindGustDir** - The direction of the strongest wind gust in the 24 hours to midnight")
        st.markdown("* **WindGustSpeed** - The speed (km/h) of the strongest wind gust in the 24 hours to midnight")
        st.markdown("* **WindDir9am** - Direction of the wind at 9am")
        st.markdown("* **WindDir3pm** - Direction of the wind at 3pm")
        st.markdown("* **WindSpeed9am** - Wind speed (km/hr) averaged over 10 minutes prior to 9am")
        st.markdown("* **WindSpeed3pm** - Wind speed (km/hr) averaged over 10 minutes prior to 3pm")
        st.markdown("* **Humidity9am** - Humidity (percent) at 9am")
        st.markdown("* **Humidity3pm** - Humidity (percent) at 3pm")
        st.markdown("* **Pressure9am** - Atmospheric pressure (hpa) reduced to mean sea level at 9am")
        st.markdown("* **Pressure3pm** - Atmospheric pressure (hpa) reduced to mean sea level at 3pm")
        st.markdown("* **Temp9am** - Temperature (degrees C) at 9am")
        st.markdown("* **Temp3pm** - Temperature (degrees C) at 3pm")
        st.markdown("* **RainToday** - Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0")
        st.markdown("* **Year, Month , Day** - The date of observation")



    
with data : 
    st.header("The Rain Dataset")
    df = pd.read_csv("X_test Dataset")
    st.write(df.head())
    
    
MinTemp = st.sidebar.slider("Please choose input for MinTemp variable" , min_value = -8.5 , max_value = 33.9 , step = 0.1)
MaxTemp = st.sidebar.slider("Please choose input for MaxTemp variable" , min_value = -4.8 , max_value = 48.1 , step = 0.1)
Rainfall = st.sidebar.slider("Please choose input for Rainfall variable" , min_value = 0.0 , max_value = 371.0 , step = 0.5)
st.sidebar.markdown("0 - North : 1-South : 2 - East : 3 - West : 4 - Others")
WindGustDir = st.sidebar.selectbox("Please choose input for WindGustDir variable" , options = [0,1,2,3,4] , index = 0)
WindDir9am = st.sidebar.selectbox("Please choose input for WindDir9am variable" , options = [0,1,2,3,4] , index = 0)
WindDir3pm = st.sidebar.selectbox("Please choose input for WindDir3pm variable" , options = [0,1,2,3,4] , index = 0)
WindGustSpeed = st.sidebar.slider("Please choose input for WindGustSpeed variable" , min_value = 6.0 , max_value = 135.0 , step = 1.0)
WindSpeed9am = st.sidebar.slider("Please choose input for WindSpeed9am variable" , min_value = 0.0 , max_value = 130.0 , step = 1.0)
WindSpeed3pm = st.sidebar.slider("Please choose input for WindSpeed3pm variable" , min_value = 0.0 , max_value = 87.0 , step = 1.0)
Humidity9am = st.sidebar.slider("Please choose input for Humidity9am variable" , min_value = 0.0 , max_value = 87.0 , step = 1.0)
Humidity3pm = st.sidebar.slider("Please choose input for Humidity3pm variable" , min_value = 0.0 , max_value = 100.0 , step = 1.0)
Pressure9am = st.sidebar.slider("Please choose input for Pressure9am variable" , min_value = 980.5 , max_value = 1041.0 , step = 0.5)
Pressure3pm = st.sidebar.slider("Please choose input for Pressure3pm variable" , min_value = 977.1 , max_value = 1039.6 , step = 0.5)
Temp9am = st.sidebar.slider("Please choose input for Temp9am variable" , min_value = -7.2 , max_value = 40.2 , step = 0.1)
Temp3pm = st.sidebar.slider("Please choose input for Temp3pm variable" , min_value = -5.4 , max_value = 46.7 , step = 0.1)
st.sidebar.markdown("0 - No : 1 - Yes")
RainToday = st.sidebar.selectbox("Please choose input for RainToday variable" , options = [0,1] , index = 0)
Year = st.sidebar.slider("Please choose input for Year variable" , min_value = 2007 , max_value = 2022 , step = 1)
Month = st.sidebar.slider("Please choose input for Month variable" , min_value = 1 , max_value = 12 , step = 1)
Day = st.sidebar.slider("Please choose input for Day variable" , min_value = 1 , max_value = 31 , step = 1)

with X_test : 
    st.header("You have entered these inputs")
    data = pd.DataFrame(data = {"MinTemp" :[MinTemp] , "MaxTemp" : [MaxTemp] , "Rainfall" : [Rainfall],
                            "WindGustDir" :[WindGustDir], "WindGustSpeed" : [WindGustSpeed] , "WindDir9am" : [WindDir9am],
                            "WindDir3pm" : [WindDir3pm] , "WindSpeed9am" : [WindSpeed9am] , "WindSpeed3pm" : [WindSpeed3pm],
     "Humidity9am" : [Humidity9am] , "Humidity3pm" : [Humidity3pm] , "Pressure9am" : [Pressure9am], "Pressure3pm" : [Pressure3pm],
    "Temp9am" : [Temp9am] , "Temp3pm" : [Temp3pm] , "RainToday" : [RainToday] , "Year" : [Year] , "Month" : [Month],"Day" : [Day]})
    data


btn_predict = st.button("Predict")

with modeling :
    if btn_predict :
        scaler = pickle.load(open("Scaler" , "rb"))
        model = tf.keras.models.load_model("Neural_Network.h5")
        y_pred_proba = tf.squeeze(model.predict(scaler.transform(pd.DataFrame(data = {"MinTemp" :[MinTemp] , "MaxTemp" : [MaxTemp] , "Rainfall" : [Rainfall],
                            "WindGustDir" :[WindGustDir], "WindGustSpeed" : [WindGustSpeed] , "WindDir9am" : [WindDir9am],
                            "WindDir3pm" : [WindDir3pm] , "WindSpeed9am" : [WindSpeed9am] , "WindSpeed3pm" : [WindSpeed3pm],
     "Humidity9am" : [Humidity9am] , "Humidity3pm" : [Humidity3pm] , "Pressure9am" : [Pressure9am], "Pressure3pm" : [Pressure3pm],
    "Temp9am" : [Temp9am] , "Temp3pm" : [Temp3pm] , "RainToday" : [RainToday] , "Year" : [Year] , "Month" : [Month],"Day" : [Day]}))))
    
       
        y_pred = tf.math.round(model.predict(scaler.transform(pd.DataFrame(data = {"MinTemp" :[MinTemp] , "MaxTemp" : [MaxTemp] , "Rainfall" : [Rainfall],
                            "WindGustDir" :[WindGustDir], "WindGustSpeed" : [WindGustSpeed] , "WindDir9am" : [WindDir9am],
                            "WindDir3pm" : [WindDir3pm] , "WindSpeed9am" : [WindSpeed9am] , "WindSpeed3pm" : [WindSpeed3pm],
     "Humidity9am" : [Humidity9am] , "Humidity3pm" : [Humidity3pm] , "Pressure9am" : [Pressure9am], "Pressure3pm" : [Pressure3pm],
    "Temp9am" : [Temp9am] , "Temp3pm" : [Temp3pm] , "RainToday" : [RainToday] , "Year" : [Year] , "Month" : [Month],"Day" : [Day]}))))
        
        
        if y_pred == [0] : 
            st.markdown("It **won't** rain tomorrow and probablity is {:.0%}".format(y_pred_proba))
            
        elif y_pred == [1] : 
            st.markdown("It **will** rain tomorrow and probablity is {:.0%}".format(y_pred_proba))
