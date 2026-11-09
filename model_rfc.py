import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Iris Prediction
         
Iris Flower Predictions Dashboard 
(use Random Forest Classifier)""")

st.image("https://content.codecademy.com/programs/machine-learning/k-means/iris.svg", caption="Iris Image", use_column_width=True)
#col1, col2, col3, col4, col5= st.columns(5, gap = 'medium')
#col5.markdown("*[Source Picture](https://content.codecademy.com/programs/machine-learning/k-means/iris.svg)*")

st.sidebar.header('Slide the Input') #user input

def user_input():
    se_lth = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    se_wth = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    pe_lth = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    pe_wth = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': se_lth,
            'sepal_width': se_wth,
            'petal_length': pe_lth,
            'petal_width': pe_wth}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()

st.subheader('Parameter Input')

col1, col2, col3, col4 = st.columns(4, gap = 'small')
col1.metric(label = '**Sepal Length**',value = df['sepal_length'])
col2.metric(label = '**Sepal Width**',value = df['sepal_width'])
col3.metric(label = '**Petal Length**',value = df['petal_length'])
col4.metric(label = '**Petal Width**',value = df['petal_width'])



iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader ('Iris Labels')
col1, col2, col3= st.columns(3, gap = 'large')
with col1:
    st.image("setosa.jpg", caption=iris.target_names[0], use_column_width=True)
    #col1, col2, col3, col4, col5= st.columns(5, gap = 'medium')
    #col5.markdown("*[Source Picture](https://www.plant-world-seeds.com/store/view_seed_item/6010/iris-setosa-baby-blue-seeds)*")
with col2:
    st.image("versicolore.jpg", caption=iris.target_names[1], use_column_width=True)
    #col1, col2, col3, col4, col5= st.columns(5, gap = 'medium')
    #col5.markdown("*[Source Picture](https://www.latour-marliac.com/en/other-plants/aquatic-shallow-water/50-iris-versicolor-iris-versicolore.html)*")
    
with col3:
    st.image("virginica.jpg", caption=iris.target_names[2], use_column_width=True)
    #col1, col2, col3, col4, col5= st.columns(5, gap = 'medium')
    #col5.markdown("*[Source Picture](https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/iris_virginica.shtml)*")
    





st.subheader ('Prediction')

if iris.target_names[prediction] == 'setosa':
    st.write('Iris Setosa')
    st.image("setosa.jpg", use_column_width=False, width=200)
elif iris.target_names[prediction] == 'versicolor':
    st.write('Iris Versicolor')
    st.image("versicolore.jpg", use_column_width=False, width=200)
else:
    st.write('Iris Virginica')
    st.image("virginica.jpg", use_column_width=False, width=200)

    

st.subheader ('Prediction Probability Value')

col1, col2, col3 = st.columns(3, gap = 'small')
col1.metric(label = '**Setosa**',value = prediction_proba[0, 0])
col2.metric(label = '**Versicolor**',value = prediction_proba[0, 1])
col3.metric(label = '**Virginica**',value = prediction_proba[0, 2])





