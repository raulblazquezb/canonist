import streamlit as st
from PIL import Image
import numpy as np
from cnn import load_model_weights
import torchvision
from cnn import CNN
from cnn import load_data
import torch
import torchvision.transforms as transforms

# Cargar la red neuronal entrenada (reemplaza este código con tu implementación)
def predecir_parte_casa(imagen):
    # Aquí deberías incluir tu código para predecir la parte de la casa basada en la imagen
    # Esta función debería devolver la parte de la casa predicha (por ejemplo, 'cocina', 'baño', etc.)
    # Por ahora, solo devolveremos un resultado aleatorio como ejemplo
    # Load model
    train_dir = 'dataset/training'
    valid_dir = 'dataset/validation'

    train_loader, valid_loader, num_classes = load_data(train_dir,
                                                        valid_dir,
                                                        batch_size=32,
                                                        img_size=224)  # ResNet50 requires 224x224 images
    model_weights = load_model_weights('resnet50-1epoch')
    model = CNN(torchvision.models.resnet50(weights=model_weights), num_classes)
    model.eval()



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

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(image)

    # Obtener la clase predicha
    _, predicted = torch.max(outputs, 1)
    classes = predicted.item()
    classnames = train_loader.dataset.classes
    return classnames[classes]

st.write("# Test App")
#st.write("## Ejemplo de app de streamlit")
