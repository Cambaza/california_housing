import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# California House Price Prediction App
This app predicts the **California House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
california = datasets.fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)[:1000]
X = X[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']]
Y = pd.DataFrame(california.target, columns=["MEDV"])[:1000]


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    MedInc= st.sidebar.slider('MedInc', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
    HouseAge = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
    AveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
    AveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
    Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
    AveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
    data = {'MedInc': MedInc, 
    'HouseAge': HouseAge, 'AveRooms': AveRooms, 'AveBedrms': AveBedrms,
    'Population': Population, 'AveOccup': AveOccup}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')

plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
st.pyplot(bbox_inches='tight')