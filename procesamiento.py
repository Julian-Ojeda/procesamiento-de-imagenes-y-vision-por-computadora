import numpy as np
from PIL import Image, ImageTk

def cargar_imagen(ruta_imagen, ancho_raw=None, alto_raw=None)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Carga una imagen (.jpg o .raw) y devuelve los arrays R, G y B.
        Si es .raw, los tres canales devuelven la misma matriz (escala de grises).

        Parámetros:
        ruta_imagen (str): ruta del archivo de imagen
        ancho_raw (int): ancho de la imagen raw
        alto_raw (int): alto de la imagen raw
        ambos por defecto None, solo necesarios para archivos .raw

        Retorna:
        objeto: tupla (R, G, B) con matrices 2D de cada canal
    """
    # SI ES ARCHIVO RAW
    if ruta_imagen.lower().endswith('.raw'):
        matriz_1d = np.fromfile(ruta_imagen, dtype=np.uint8)
        gris = matriz_1d.reshape((alto_raw, ancho_raw))
        return gris, gris, gris
        
    # SI ES JPG / OTRO FORMATO
    else:
        
        img = Image.open(ruta_imagen).convert("RGB")
        matriz = np.array(img)
        return matriz[:, :, 0], matriz[:, :, 1], matriz[:, :, 2]
    
def engrisar(R, G, B)-> np.ndarray:
    """
    Convierte los canales R, G y B a una imagen en escala de grises.

    Parámetros:
    R, G, B (numpy.ndarray): matrices 2D con intensidades de cada canal

    Retorna:
    numpy.ndarray: matriz 2D con la imagen en escala de grises
    """
    # Usamos la fórmula estándar para convertir a gris
    gris = 0.299 * R + 0.587 * G + 0.114 * B
    return gris.astype(np.uint8)


def guardar_imagen_gris(gris, ruta_salida)-> None:
    """
    Guarda una imagen en escala de grises a un archivo.

    Parámetros:
    gris (numpy.ndarray): matriz 2D con la imagen en escala de grises
    ruta_salida (str): ruta del archivo donde se guardará la imagen
    """
    img_gris = Image.fromarray(gris)
    img_gris.save(ruta_salida)

def padding(imagen, alto_k, ancho_k)-> np.ndarray:  
    """
    Crea un margen perimetral alrededor de la imagen duplicando sus bordes y esquinas.
    Devuelve la matriz ampliada.

    esto solo lo uso dentro de la ventana deslizante porque depende de variables que se declaran
    recien cuando tengo el operador (kernel) listo, y necesito sus dimensiones para calcular el margen.
    """
    alto_img, ancho_img = imagen.shape
    
    # Calcular el margen (cuántos píxeles sobresale el operador desde su centro)
    margen_y = alto_k // 2
    margen_x = ancho_k // 2
    
    # Crear matriz contenedora vacía (llena de ceros)
    alto_pad = alto_img + (2 * margen_y)
    ancho_pad = ancho_img + (2 * margen_x)
    imagen_pad = np.zeros((alto_pad, ancho_pad), dtype=np.float32)
    
    # Paso A: Copiar la imagen original en el centro
    imagen_pad[margen_y : margen_y + alto_img, margen_x : margen_x + ancho_img] = imagen
    
    # Paso B: Replicar las filas (superior e inferior) hacia el margen exterior
    for i in range(margen_y):
        imagen_pad[i, margen_x : margen_x + ancho_img] = imagen[0, :]
        imagen_pad[alto_pad - 1 - i, margen_x : margen_x + ancho_img] = imagen[-1, :]
        
    # Paso C: Replicar las columnas (izquierda y derecha) hacia el margen exterior
    for j in range(margen_x):
        imagen_pad[margen_y : margen_y + alto_img, j] = imagen[:, 0]
        imagen_pad[margen_y : margen_y + alto_img, ancho_pad - 1 - j] = imagen[:, -1]
        
    # Paso D: Replicar las 4 esquinas usando los píxeles de los vértices originales
    imagen_pad[0:margen_y, 0:margen_x] = imagen[0, 0]
    imagen_pad[0:margen_y, ancho_pad-margen_x:] = imagen[0, -1]
    imagen_pad[alto_pad-margen_y:, 0:margen_x] = imagen[-1, 0]
    imagen_pad[alto_pad-margen_y:, ancho_pad-margen_x:] = imagen[-1, -1]
    
    return imagen_pad

def ventana_deslizante(imagen, operador, progress_callback=None)-> np.ndarray:
    """
    Aplica un operador (kernel) a una imagen 2D usando una ventana deslizante.
    Retorna la matriz resultante en float32 (normalizada a 0-255).

    Parámetros:
    imagen (numpy.ndarray): matriz 2D con la imagen a procesar  
    operador (numpy.ndarray): matriz 2D con el kernel a aplicar
    progress_callback (callable, opcional): función que recibe un porcentaje de progreso.
    
    Retorna:
    numpy.ndarray: en formato unit.8, con valores normalizados entre 0 y 255
    """
    alto_img, ancho_img = imagen.shape
    alto_k, ancho_k = operador.shape
    
    # Llamamos a nuestra función de padding
    imagen_pad = padding(imagen, alto_k, ancho_k)
    
    imagen_salida = np.zeros_like(imagen, dtype=np.float32)
    
    pasos = max(1, alto_img // 50)
    for y in range(alto_img):
        for x in range(ancho_img):
            ventana = imagen_pad[y : y + alto_k, x : x + ancho_k]
            imagen_salida[y, x] = np.sum(ventana * operador)

        if progress_callback is not None and (y % pasos == 0 or y == alto_img - 1):
            progress_callback(int((y + 1) / alto_img * 100))

    imagen_salida = normalizar(imagen_salida)            
            
    return imagen_salida
    
def normalizar(imagen, out_min=0, out_max=255)-> np.ndarray:
    """
    Normaliza una imagen a un rango dado.

    Parámetros:
    imagen (numpy.ndarray): matriz 2D con valores a normalizar
    out_min (int): valor mínimo del rango de salida
    out_max (int): valor máximo del rango de salida

    Retorna:
    numpy.ndarray: matriz 2D con valores normalizados en el rango [out_min, out_max]
    """
    min_val = np.min(imagen)
    max_val = np.max(imagen)
    
    if max_val - min_val == 0:
        return np.full_like(imagen, out_min, dtype=np.uint8)
    
    imagen_normalizada = ((imagen - min_val) * (out_max - out_min) / (max_val - min_val)) + out_min
    return np.clip(imagen_normalizada, out_min, out_max).astype(np.uint8)