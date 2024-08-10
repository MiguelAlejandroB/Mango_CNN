import cv2 #Modulo de la libreria opencv-python
import numpy as np
import os

def Redimencionar_imagen(ruta_imagen):
    # Cargar la imagen
    image = cv2.imread(ruta_imagen)

    # Redimensionar la imagen a 100x100 píxeles
    resized_image = cv2.resize(image, (100, 100))

    # Guardar la imagen redimensionada (opcional)
    #cv2.imwrite('ruta_a_la_imagen_redimensionada.jpg', resized_image)
    return resized_image

def Ecualizacion_del_Histograma(ruta_imagen):
    # Cargar la imagen en escala de grises
    image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    # Ecualizar el histograma
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def Operaciones_Morfologicas(ruta_imagen):
    # Cargar la imagen
    image = cv2.imread(ruta_imagen)

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


def MatrizColor_RGB(ruta_imagen):
    # Cargar la imagen
    image = cv2.imread(ruta_imagen)

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def redimensionar_imagenes_en_carpeta(ruta_carpeta, ruta_salida):
    # Crear la carpeta de salida si no existe
    os.makedirs(ruta_salida, exist_ok=True)

    # Obtener todos los archivos en la carpeta
    archivos = os.listdir(ruta_carpeta)

    # Filtrar solo los archivos .jpg
    archivos_jpg = [archivo for archivo in archivos if archivo.lower().endswith('.jpg')]

    for archivo in archivos_jpg:
        # Construir la ruta completa del archivo
        ruta_imagen = os.path.join(ruta_carpeta, archivo)

        # Redimensionar la imagen usando la función definida
        imagen_redimensionada = Redimencionar_imagen(ruta_imagen)

        if imagen_redimensionada is None:
            continue

        # Construir el nombre del archivo redimensionado
        nombre_archivo, extension = os.path.splitext(archivo)
        nuevo_nombre = f"{nombre_archivo}_escalado.jpg"
        ruta_guardar = os.path.join(ruta_salida, nuevo_nombre)

        # Guardar la imagen redimensionada
        cv2.imwrite(ruta_guardar, imagen_redimensionada)

        print(f"Imagen guardada como {nuevo_nombre} en {ruta_salida}")


def procesar_imagenes_en_carpeta(ruta_carpeta, ruta_salida):
    # Crear carpetas para cada tipo de pre-proceso si no existen
    ruta_ecualizada = os.path.join(ruta_salida, 'Ecualizada_red')
    ruta_morfologica = os.path.join(ruta_salida, 'Morfologica_red')
    ruta_rgb = os.path.join(ruta_salida, 'RGB_red')

    os.makedirs(ruta_ecualizada, exist_ok=True)
    os.makedirs(ruta_morfologica, exist_ok=True)
    os.makedirs(ruta_rgb, exist_ok=True)

    # Obtener todos los archivos en la carpeta
    archivos = os.listdir(ruta_carpeta)

    # Filtrar solo los archivos .jpg
    archivos_jpg = [archivo for archivo in archivos if archivo.lower().endswith('.jpg')]

    if not archivos_jpg:
        print("No se encontraron archivos .jpg en la carpeta.")
        return

    for archivo in archivos_jpg:
        # Construir la ruta completa del archivo
        ruta_imagen = os.path.join(ruta_carpeta, archivo)

        # Procesar la imagen con cada método
        imagen_ecualizada = Ecualizacion_del_Histograma(ruta_imagen)
        imagen_morfologica = Operaciones_Morfologicas(ruta_imagen)
        imagen_rgb = MatrizColor_RGB(ruta_imagen)

        # Guardar las imágenes procesadas
        if imagen_ecualizada is not None:
            nombre_archivo, _ = os.path.splitext(archivo)
            cv2.imwrite(os.path.join(ruta_ecualizada, f'{nombre_archivo}_ecualizada.jpg'), imagen_ecualizada)
            print(f"Imagen ecualizada guardada como {nombre_archivo}_ecualizada.jpg en {ruta_ecualizada}")

        if imagen_morfologica is not None:
            cv2.imwrite(os.path.join(ruta_morfologica, f'{nombre_archivo}_morfologica.jpg'), imagen_morfologica)
            print(f"Imagen morfológica guardada como {nombre_archivo}_morfologica.jpg en {ruta_morfologica}")

        if imagen_rgb is not None:
            cv2.imwrite(os.path.join(ruta_rgb, f'{nombre_archivo}_rgb.jpg'), imagen_rgb)
            print(f"Imagen RGB guardada como {nombre_archivo}_rgb.jpg en {ruta_rgb}")

# Rutas específicas
#ruta_carpeta = r'C:\Users\User\OneDrive - Universidad Nacional de Colombia\Documentos\Mangos\Datos de mangos\Mango Red'
#ruta_salida = r'C:\Users\User\OneDrive - Universidad Nacional de Colombia\Documentos\Mangos\Datos de mangos'

# Ejecutar la función
#redimensionar_imagenes_en_carpeta(ruta_carpeta, ruta_salida)


#procesar_imagenes_en_carpeta(ruta_carpeta, ruta_salida)