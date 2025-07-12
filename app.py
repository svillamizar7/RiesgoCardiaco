# prompt: Eres experto en streamlit y requiero realizar un deployment de la aplicación para determinar si un paciente sufrirá o no del corazón. El modelo fue entrenado usando sklearn con SVC y los datos de entrada fueron escalados usando minmax scaler. Modelo: modelo_svc.jb Escalador: scaler.jb Los modelos fueron guardados usando joblib Coloque un título así: Modelo IA para predicción de problemas cardiacos Haga un resumen de cómo funciona el modelo para los usuarios. Coloque en la parte de abajo "Elaborado por: SergioVilla" con un emoji de copyright UNAB 2025 En el lado izquierdo coloca un sidebar donde con un slider el usuario escoja lo siguiente: Edad: 20 años a 80 años con incrementos de 1 año. Colesterol: Use los valores de parámetros de niveles de colesterol desde 120 hasta 600 con incrementos de 10. Por defecto, los valores seleccionados sean Edad: 20 años, Colesterol: 200. Estos datos deben pasar por el scaler. Los resultados son: 0: No sufrirá del corazón, ponerlo en fondo verde y letras negras con un emoji feliz y debajo aparece una imagen llamada NoSufre.jpg; y 1: Sufrirá del corazón, ponerlo en fondo rojo con letras negras y un emoji triste, y debajo una imagen llamada SiSufre.jpg. Antes del título poner una imagen tipo banner llamada Cabezote.jpg
# NO EJECUTAR CON EL BOTÓN DE RUN
# Se ejecuta primero cargando las bibliotecas requeridas con pip install -r requirements.txt
# Se ejecuta con streamlit run app.py desde el terminal

import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the model and scaler
try:
    modelo_svc = joblib.load('modelo_svc.jb')
    scaler = joblib.load('scaler.jb')
except FileNotFoundError:
    st.error("Error: modelo_svc.jb or scaler.jb not found. Please ensure they are in the same directory as the app.")
    st.stop()

# Load images
try:
    cabezote_img = Image.open('Cabezote.jpg')
    no_sufre_img = Image.open('NoSufre.jpg')
    si_sufre_img = Image.open('SiSufre.jpg')
except FileNotFoundError:
    st.warning("Warning: One or more image files (Cabezote.jpg, NoSufre.jpg, SiSufre.jpg) not found. Image display will be skipped.")
    cabezote_img = None
    no_sufre_img = None
    si_sufre_img = None

# Set page title and icon
st.set_page_config(page_title="Modelo IA para predicción de problemas cardiacos", page_icon=":heart:")

# Display banner image
if cabezote_img:
    #st.image(cabezote_img, use_column_width=True)
    st.image(cabezote_img, use_container_width=True)

# Title
st.title("Modelo IA para predicción de problemas cardiacos")

# Model explanation
st.write("""
Este modelo utiliza Machine Learning para predecir la probabilidad de que un paciente sufra problemas cardiacos
basado en su edad y nivel de colesterol. El modelo fue entrenado utilizando un algoritmo de Máquinas de Vectores de Soporte (SVC)
y los datos de entrada fueron escalados para mejorar el rendimiento del modelo.
""")

# Sidebar for user input
st.sidebar.header("Parámetros del paciente")

edad = st.sidebar.slider("Edad", 20, 80, 20, 1)
colesterol = st.sidebar.slider("Colesterol", 120, 600, 200, 10)

# Prepare input data for prediction
input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Scale the input data
scaled_input_data = scaler.transform(input_data)
scaled_input_df = pd.DataFrame(scaled_input_data, columns=['edad', 'colesterol'])

# Make prediction
prediction = modelo_svc.predict(scaled_input_df)

# Display results
st.subheader("Resultado de la Predicción:")

if prediction[0] == 0:
    st.markdown("<div style='background-color: lightgreen; padding: 10px; border-radius: 5px; color: black;'>", unsafe_allow_html=True)
    st.write("## **0: ¡No sufrirá del corazón! 😊**")
    st.markdown("</div>", unsafe_allow_html=True)
    if no_sufre_img:
        #st.image(no_sufre_img, use_column_width=False, width=300)
        st.image(no_sufre_img, use_container_width=False, width=300)
else:
    st.markdown("<div style='background-color: lightcoral; padding: 10px; border-radius: 5px; color: black;'>", unsafe_allow_html=True)
    st.write("## **1: Sufrirá del corazón 😥**")
    st.markdown("</div>", unsafe_allow_html=True)
    if si_sufre_img:
        #st.image(si_sufre_img, use_column_width=False, width=300)
        st.image(si_sufre_img, use_container_width=False, width=300)

# Footer
st.markdown("---")
st.write("Elaborado por: SergioVilla © UNAB 2025")
