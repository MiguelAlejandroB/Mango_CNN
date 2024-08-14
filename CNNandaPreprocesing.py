import cv2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd
def Redimencionar_imagen(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Redimensionar la imagen a 100x100 píxeles
    resized_image = cv2.resize(image, (100, 100))

    # Guardar la imagen redimensionada (opcional)
    #cv2.imwrite('ruta_a_la_imagen_redimensionada.jpg', resized_image)
    return resized_image

def Ecualizacion_del_Histograma(image_path):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ecualizar el histograma
    equalized_image = cv2.equalizeHist(image)

    return equalized_image
def redimen(image):
          # Redimensionar la imagen a 100x100 píxeles
    image = cv2.resize(image, (100, 100))
    return image
def Operaciones_Morfologicas(image_path):
        # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un umbral para obtener una imagen binaria
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Definir un kernel para las operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)

    # Aplicar la operación de dilatación
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Aplicar la operación de erosión
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Encontrar los contornos
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 1)

    # Crear una imagen en color para superponer resultados
    combined_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Superponer la imagen binaria
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    combined_result = cv2.addWeighted(combined_result, 0.5, binary_colored, 0.5, 0)

    # Superponer la imagen erosionada
    eroded_colored = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    combined_result = cv2.addWeighted(combined_result, 0.5, eroded_colored, 0.5, 0)

    # Superponer los contornos en la imagen original
    combined_result = cv2.addWeighted(combined_result, 0.7, contoured_image, 0.3, 0)

    return combined_result


def MatrizColor_RGB(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def convert_to_color(image_array):
    # Asumiendo que image_array es en escala de grises
    color_image = np.stack([image_array]*3, axis=-1)
    return color_image

def load_image(imgn):
    return np.array(imgn) / 255.0  # Normalizar entre 0 y 1

def allProces(image_path):
    image_scale = Redimencionar_imagen(image_path)
    image_equalized = Ecualizacion_del_Histograma(image_path)
    image_morfolo = Operaciones_Morfologicas(image_path)
    image_RGB = MatrizColor_RGB(image_path)

    image_scale = redimen(image_scale)
    image_equalized = redimen(image_equalized)
    image_equalized = convert_to_color(image_equalized)
    image_morfolo = redimen(image_morfolo)
    image_RGB = redimen(image_RGB)
    print(image_scale.shape)
    print(image_equalized.shape)
    print(image_morfolo.shape)
    print(image_RGB.shape)
    x_data=[]
    x = np.stack([image_scale,image_RGB, image_morfolo, image_equalized], axis=-1)
    x_data.append(x)
    return x_data
def runCNN(x_data):
    # Cargar el modelo
    model = tf.keras.models.load_model('/content/drive/MyDrive/Mango/CNNs_Mango.h5')
    # Hacer predicciones con el modelo
    predictions = model.predict(x_data)
    dit ={'prediccion':predictions.flatten()}
    # Crear el DataFrame
    print(dit)

x_data=allProces('/content/drive/MyDrive/Mango/Test/Captura.PNG')
x_data=np.array(x_data)
print("Tamaño de x_data:", x_data.shape)
runCNN(x_data)