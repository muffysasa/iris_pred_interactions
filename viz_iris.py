import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

st.write("""
# Iris Visualization""")

iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['species'] = iris.target_names[iris.target]
categories = iris_df['species'].value_counts().reset_index()

st.write('### Total Data per Species')
col1, col2, col3 = st.columns(3, gap = 'small')
col1.metric(label = '**Setosa**',value = categories['count'].loc[0])
col2.metric(label = '**Versicolor**',value = categories['count'].loc[1])
col3.metric(label = '**Virginica**',value = categories['count'].loc[2])

st.subheader('Range rata-rata lebar dan panjang setiap spesies')
col1, col2= st.columns(2, gap = 'large')
with col1:
    st.text('Range rata rata pada Sepal Length Iris')
    fig, ax = plt.subplots(figsize=(16,8))
    pallete = {'setosa': 'lightblue', 'versicolor': 'lightgreen', 'virginica': 'lightcoral'}
    plot = sns.boxplot(x='species', y='sepal length (cm)', data=iris_df, palette=pallete)
    st.pyplot(plot.figure)
with col2:
    st.text('Range rata rata pada Sepal Width Iris')
    fig, ax = plt.subplots(figsize=(16, 8))
    pallete = {'setosa': 'lightblue', 'versicolor': 'lightgreen', 'virginica': 'lightcoral'}
    plot = sns.boxplot(x='species', y='sepal width (cm)', data=iris_df, palette=pallete)
    st.pyplot(plot.figure)
    

    
col1, col2= st.columns(2, gap = 'large')
with col1:
    st.text('Range rata rata pada Petal Length Iris')
    fig, ax = plt.subplots(figsize=(16, 8))
    pallete = {'setosa': 'lightblue', 'versicolor': 'lightgreen', 'virginica': 'lightcoral'}
    plot = sns.boxplot(x='species', y='petal length (cm)', data=iris_df, palette=pallete)
    st.pyplot(plot.figure)
with col2:
    st.text('Range rata rata pada Petal Width Iris')
    fig, ax = plt.subplots(figsize=(16, 8))
    pallete = {'setosa': 'lightblue', 'versicolor': 'lightgreen', 'virginica': 'lightcoral'}
    plot = sns.boxplot(x='species', y='petal width (cm)', data=iris_df, palette=pallete)
    st.pyplot(plot.figure)

st.subheader('Pola hubungan lebar dan panjang setiap spesies')
fig, ax = plt.subplots(figsize=(16, 8))
pallete = {'setosa': 'navy', 'versicolor': 'green', 'virginica': 'red'}
plot = sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df, palette=pallete)
st.pyplot(plot.figure)

fig, ax = plt.subplots(figsize=(16, 8))
pallete = {'setosa': 'navy', 'versicolor': 'green', 'virginica': 'red'}
plot = sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=iris_df, palette=pallete)
st.pyplot(plot.figure)





 


