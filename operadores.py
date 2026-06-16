import numpy as np
from PIL import Image   
import procesamiento as p
import ruidos as r
import filtros as f
import matplotlib.pyplot as plt


def negativo(imagen)-> np.ndarray:
    return 255 - imagen

def histograma(imagen):
    hist = np.zeros(256, dtype=int)
    for pixel in imagen.flatten():
        hist[pixel] += 1
    return hist

def umbral(imagen, umbral):
    return (imagen >= umbral).astype(np.uint8) * 255

def ecualizacion_histograma(imagen):

    hist = np.zeros(256, dtype=int)
    for pixel in imagen.flatten():
        hist[pixel] += 1
    fda = np.cumsum(hist / imagen.size)
    fda_min = fda[fda > 0].min()
    sk = np.round((fda - fda_min) / (1 - fda_min) * 255).astype(np.uint8)
    return sk[imagen]

def umbralizacion(imagen, umbral):

    imagen_umbralizada = np.where(
        imagen >= umbral,
        255,
        0
    )

    return imagen_umbralizada.astype(np.uint8)