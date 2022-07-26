import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
#with open('custom.css') as f:        
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
st.markdown("<h1 style='text-align: center;'>✿ Iris Flower Prediction App ✿</h1>", unsafe_allow_html=True)

#st.sidebar.image('setosa.png', width= 225)
st.sidebar.table(pd.DataFrame({'Species/Type': ['Iris Setosa','Iris Versicolor','Iris Virginica'],}))
st.sidebar.markdown("<h2 style='text-align: center;'>User Input Parameters</h2>", unsafe_allow_html=True)

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'Sepal Length': sepal_length,
            'Sepal Width': sepal_width,
            'Petal Length': petal_length,
            'Petal Width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

iris = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/iris/main/IRIS.csv')
X = iris.drop('species', axis=1)
Y = iris.species

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

col1, col2, col3 = st.columns([2,4,2])   
 
with col1:
   st.write(' ')

with col2:
   st.image('flowers.png', width=500)

with col3:
   st.write(' ')

tab1, tab2 = st.tabs(["User Input Parameters", "Results"])

with tab1:
    st.markdown('Input in the table will change based on what user pick.')
    st.write(df)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Prediction')
        st.write(prediction)
    
    with col2:
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        
     
     
     
        
        
        
        
   

    
