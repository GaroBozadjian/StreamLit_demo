import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

#the header of the app that shows you what is the app for
st.write("""
# Penguin Prediction App

This app predicts penguin species
""")

# The sidebar of the app were all the features we need to choose to predict the species of the penguins
st.sidebar.header('User Input Features')

# if you have a data you need to predict the species you can upload it the csv file
# else input your own values for the features to predict the species
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_f../StreamLit_demo/eatures()

# read the original that we train the model with, encode it then add the parameters you added in the side bar to it
penguins_raw=pd.read_csv('penguins_cleaned.csv')
penguins=penguins_raw.drop('species',axis=1)
df=pd.concat([input_df,penguins],axis=0)

encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
df=df[:1]

#shows your input paramters you added either with uploading a csv or from the sidebar features
st.subheader('User input features')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# loading the saved model
load_clf=pickle.load(open('penguins_clf.pkl','rb'))

#predict the species 
prediction=load_clf.predict(df)
prediction_prob=load_clf.predict_proba(df)

# show the type of the species
st.subheader('Predicition')
penguins_species=np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species)

st.subheader('Prediction Probability')
st.write(prediction_prob)