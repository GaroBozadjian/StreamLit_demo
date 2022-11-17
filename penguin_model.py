import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read the data
pen=pd.read_csv('/home/garo/Desktop/streamlit/penguins_cleaned.csv')

df=pen.copy()

# Target and the columns that should be encoded
target='species'
encode=['sex','island']

# encode the columns
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]

# map the target
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# slpit the data to X and Y 
X=df.drop('species',axis=1)
Y=df['species']

# Fit the model 
clf=RandomForestClassifier()
clf.fit(X,Y)

# save the model
pickle.dump(clf,open('/home/garo/Desktop/streamlit/penguins_clf.pkl','wb'))