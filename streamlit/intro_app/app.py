import streamlit as st
from PIL import Image
import numpy as np

# Cargar la red neuronal entrenada (reemplaza este código con tu implementación)
def predecir_parte_casa(imagen):
    # Aquí deberías incluir tu código para predecir la parte de la casa basada en la imagen
    # Esta función debería devolver la parte de la casa predicha (por ejemplo, 'cocina', 'baño', etc.)
    # Por ahora, solo devolveremos un resultado aleatorio como ejemplo
    partes_posibles = ['cocina', 'baño', 'dormitorio', 'sala de estar']
    return np.random.choice(partes_posibles)

# Configuración de la aplicación
st.set_page_config(layout="wide", page_title="Predicción de Partes de la Casa")

# Título de la aplicación
st.title("Predicción de Partes de la Casa")

# Subtítulo e instrucciones
st.write("Por favor, carga una imagen de una parte de una casa y haremos una predicción de qué parte es.")

# Cargar la imagen
imagen = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

# Si se ha cargado una imagen
if imagen is not None:
    # Mostrar la imagen cargada
    st.image(imagen, caption='Imagen cargada', use_column_width=True)

    # Procesar la imagen y realizar la predicción
    imagen_pil = Image.open(imagen)
    parte_predicha = predecir_parte_casa(imagen_pil)

    # Mostrar la predicción
    st.write(f"La parte de la casa predicha es: {parte_predicha}")