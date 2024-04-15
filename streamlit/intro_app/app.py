import streamlit as st
from PIL import Image
import numpy as np
from cnn import load_model_weights
import torchvision
from cnn import CNN
from cnn import load_data
import torch
import torchvision.transforms as transforms
import os

# Cargar la red neuronal entrenada (reemplaza este código con tu implementación)
def predecir_parte_casa(imagen):
    # Aquí deberías incluir tu código para predecir la parte de la casa basada en la imagen
    # Esta función debería devolver la parte de la casa predicha (por ejemplo, 'cocina', 'baño', etc.)
    # Por ahora, solo devolveremos un resultado aleatorio como ejemplo
    # Load model
    train_dir = '../../03TransferLearning/dataset/training'
    test_dir = '../../03TransferLearning/dataset/test'

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # NOTE
    # Guardar la imagen en disco (cambia esto para que guarde la imagen que quieras predecir)!!
    folder = "/Living room"
    imagen.save(test_dir + folder + '/test_image.png')

    train_loader, test_loader, num_classes = load_data(train_dir,
                                                        test_dir,
                                                        batch_size=32,
                                                        img_size=224)  # ResNet50 requires 224x224 images
    model_weights = load_model_weights('../../03TransferLearning/models/resnet50-10epoch_unf_5')
    model = CNN(torchvision.models.resnet50(weights=model_weights), num_classes)
    model.load_state_dict(model_weights)



    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convierte la imagen a un tensor
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza la imagen
    ])
  # Reemplaza 'tu_imagen.jpg' con la ruta de tu imagen

    # Aplicar las transformaciones a la imagen
    image = transform(imagen)

    image = torch.cat([image, image, image], dim=0)

    # Agregar una dimensión al tensor para simular un batch de tamaño 1
    image = image.unsqueeze(0)

    prediction = model.predict(test_loader)

    # Obtener la clase predicha
    classnames = train_loader.dataset.classes
    prediction = classnames[prediction[0]]
    return prediction

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