import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

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

with st.expander("User Input Parameters"):
    st.markdown('Input changes based on what user pick.')
    st.write(df)

iris = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/iris/main/IRIS.csv')
X = iris.drop('species', axis=1)
Y = iris.species

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

tab1, tab2, tab3 = st.tabs(["Class labels", "Prediction", "Prediction Probability"])

with tab1:
    st.subheader('Class labels and index number')
    st.write(pd.DataFrame({'Species': ['Iris-setosa','Iris-versicolor','Iris_virginica'],}))

with tab2:
     st.subheader('Prediction')
     st.write(prediction)

with tab3:
     st.subheader('Prediction Probability')
     st.write(prediction_proba)
        
        
        
        
   

    
