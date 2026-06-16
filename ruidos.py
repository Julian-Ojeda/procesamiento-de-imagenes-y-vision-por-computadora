import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import procesamiento as p

def gaussiana(mu, sigma, n):
    return np.random.normal(mu, sigma, n)

def exponencial(lam, n):
    return np.random.exponential(1 / lam, n)

def ruido_sal_pimienta(imagen, densidad=0.05):
    p = min(densidad, 0.49)  # p entre (0, 0.5)
    x = np.random.uniform(0, 1, imagen.shape[:2])  # tomamos un x con distribución uniforme entre (0, 1)
    img_contaminada = imagen.copy()
    img_contaminada[x <= p] = 0  # si x <= p entonces I(i,j) = 0
    img_contaminada[x > 1 - p] = 255  # si x > p entonces I(i,j) = 255
    return img_contaminada

def agregar_ruido_gaussiano(imagen, sigma)-> np.ndarray:
    """
    Suma ruido Gaussiano aditivo a una imagen.
    Parámetros:
    imagen (numpy.ndarray): matriz 2D con la imagen a la que se le agregará ruido
    sigma (float): desviación estándar del ruido Gaussiano (default=10)

    Retorna:
    numpy.ndarray: imagen con ruido agregado, normalizada a 0-255 por el meetodo de normalización de procesamiento.py   
    """
    ruido = np.random.normal(0, sigma, imagen.shape)
    imagen_ruidosa = imagen.astype(np.float32) + ruido
    imagen_ruidosa = p.normalizar(imagen_ruidosa)
    return imagen_ruidosa

def transformacion_potencia(imagen, gamma, c=1)-> np.ndarray:
    """
    Aplica una transformación de potencia (gamma) a una imagen.
    Parámetros:
    imagen (numpy.ndarray): matriz 2D con la imagen a la que se le aplicará la transformación
    gamma (float): parámetro de potencia
    c (float): constante de escala (default=1)

    Retorna:
    numpy.ndarray: imagen con transformación de potencia aplicada, normalizada a 0-255
    """
    img = imagen.astype(np.float32) / 255.0
    return p.normalizar(c * np.power(img, gamma) * 255, 0, 255)


