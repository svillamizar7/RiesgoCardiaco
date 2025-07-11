import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo y el scaler
# Asegúrate de que 'modelo_svc.pkl' esté en la misma carpeta o proporciona la ruta correcta
try:
    model_svc = joblib.load('modelo_svc.pkl')
    # Necesitamos el scaler usado para normalizar los datos de entrenamiento
    # Dado que no se guardó en el cuaderno original, asumiremos un nuevo scaler
    # y lo fitearemos con datos de ejemplo o si es posible, guardarlo en el futuro.
    # Para este ejemplo, usaremos valores de ejemplo para el scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler with example data that represents the range of your training data
    # (ideally you would save the fitted scaler)
    example_data = pd.DataFrame({'edad': [0, 100], 'colesterol': [0, 600]}) # Example range
    scaler.fit(example_data[['edad', 'colesterol']])

except FileNotFoundError:
    st.error("Error: modelo_svc.pkl no encontrado. Asegúrate de que el archivo del modelo está en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo o el scaler: {e}")
    st.stop()


st.title("Predicción de Problemas Cardiacos - Support Vector Machine (SVC)")
st.markdown("<h2 style='color: red;'>Elaborado por SergioVilla</h2>", unsafe_allow_html=True)

st.write("Por favor, introduce los valores para predecir el riesgo de problemas cardiacos.")

edad = st.slider("Edad", min_value=0, max_value=100, value=50)
colesterol = st.slider("Colesterol", min_value=0, max_value=600, value=200)

if st.button("Predecir"):
    # Crear un DataFrame con los valores de entrada
    input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

    # Normalizar los datos de entrada usando el mismo scaler que se usó para entrenar el modelo
    # Aquí asumimos que el scaler fue fitado con un rango similar al de entrenamiento
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción
    prediction = model_svc.predict(input_data_scaled)

    if prediction[0] == 1:
        st.write("Riesgo de Problemas Cardiacos 😢")
    else:
        st.write("Paciente Sano 😊")
