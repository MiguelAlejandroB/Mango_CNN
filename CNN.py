
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Función para cargar una imagen y convertirla en un array numpy
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Redimensionar a 100x100
    return np.array(img) / 255.0  # Normalizar entre 0 y 1


# Función para cargar el conjunto de datos
def load_dataset(base_dir):
    x_data = []
    y_data = []

    # Definir las categorías
    categories = ['green', 'red']

    for category in categories:
        category_path = os.path.join(base_dir, category)

        if not os.path.exists(category_path):
            print(f"Directorio no encontrado: {category_path}")
            continue

        # Obtener los nombres de los archivos sin la extensión
        files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        names = set(f.split('_')[0] for f in files)  # Nombre base de las imágenes

        for name in names:
            # Construir las rutas de las imágenes
            normal_path = os.path.join(category_path, f"{name}.jpg")
            rgb_path = os.path.join(category_path, f"{name}_rgb.jpg")
            morfologica_path = os.path.join(category_path, f"{name}_morfologica.jpg")
            ecualizada_path = os.path.join(category_path, f"{name}_ecualizada.jpg")

            # Verificar si todas las imágenes existen
            if all(os.path.exists(p) for p in [normal_path, rgb_path, morfologica_path, ecualizada_path]):
                normal_img = load_image(normal_path)
                rgb_img = load_image(rgb_path)
                morfologica_img = load_image(morfologica_path)
                ecualizada_img = load_image(ecualizada_path)

                # Apilar las imágenes en una sola matriz
                x = np.stack([normal_img, rgb_img, morfologica_img, ecualizada_img], axis=-1)
                x_data.append(x)
                y_data.append(categories.index(category))

    return np.array(x_data), np.array(y_data)

# Ruta a la carpeta principal de datos
base_dir = r'C:\Users\User\OneDrive - Universidad Nacional de Colombia\Documentos\Mangos\Datos de mangos'

# Cargar el conjunto de datos
x_data, y_data = load_dataset(base_dir)

# Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
