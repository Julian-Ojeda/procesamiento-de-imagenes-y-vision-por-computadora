import numpy as np
from PIL import Image
import procesamiento as p
import operadores as op
import ruidos as r
import detectores as det

def filtro_media(imagen, tam=3):
    return p.ventana_deslizante(imagen, np.ones((tam, tam)) / tam**2)

def aplicar_filtro_media(imagen, tam, progress_callback=None):
    kernel = np.ones((tam, tam)) / (tam * tam)
    return p.ventana_deslizante(imagen, kernel, progress_callback)

def filtro_mediana(imagen, k, progress_callback=None)-> np.ndarray:

    img_pad = p.padding(imagen, k, k) 
    
    filas, cols = imagen.shape
    resultado = np.zeros_like(imagen, dtype=np.float32)
    # ventana deslizante que tengo que codear porque no es lineal, no puedo usar la función de procesamiento.py     
    pasos = max(1, filas // 50)
    for i in range(filas):
        for j in range(cols):
            vecindad = img_pad[i:i+k, j:j+k].flatten()
            vecindad_ordenada = np.sort(vecindad)
            resultado[i, j] = vecindad_ordenada[k**2 // 2]
        if progress_callback is not None and (i % pasos == 0 or i == filas - 1):
            progress_callback(int((i + 1) / filas * 100))
    return resultado

def filtro_mediana_ponderada(imagen, k=3, progress_callback=None)-> np.ndarray:
    mitad = k // 2

    # Máscara de pesos: 1 en las esquinas, 2 en los lados y 4 en el centro.
    mascara = np.ones((k, k), dtype=int)
    mascara[mitad, :] = 2
    mascara[:, mitad] = 2
    mascara[mitad, mitad] = 4

    img_pad = p.padding(imagen, k, k)
    filas, cols = imagen.shape
    resultado = np.zeros_like(imagen, dtype=np.float32)
    pasos = max(1, filas // 50)

    for i in range(filas):
        for j in range(cols):
            ventana = img_pad[i:i+k, j:j+k]
            valores = np.repeat(ventana.flatten(), mascara.flatten())
            resultado[i, j] = np.median(valores)
        if progress_callback is not None and (i % pasos == 0 or i == filas - 1):
            progress_callback(int((i + 1) / filas * 100))

    return resultado.astype(np.uint8)


def aplicar_filtro_mediana_ponderada(imagen, k, progress_callback=None):
    return filtro_mediana_ponderada(imagen, k, progress_callback=progress_callback)

def filtro_gauss(imagen, sigma):
    tam = int(2 * sigma + 1)
    mitad = tam // 2
    mascara = np.array([[np.exp(-((i-mitad)**2 + (j-mitad)**2) / (2*sigma**2))
                         for j in range(tam)] for i in range(tam)])
    return mascara / mascara.sum()

def aplicar_filtro_gauss(imagen, sigma, progress_callback=None):
    kernel = filtro_gauss(imagen, sigma)
    return p.ventana_deslizante(imagen, kernel, progress_callback)

def filtro_definicion(imagen, tam=3):
    mitad = tam // 2
    kernel = np.zeros((tam, tam), dtype=np.float32)
    kernel[mitad, :] = -1
    kernel[:, mitad] = -1
    kernel[mitad, mitad] = 2 * tam - 1
    return kernel

def aplicar_filtro_definicion(imagen, tam, progress_callback=None):
    kernel = filtro_definicion(imagen, tam)
    return p.ventana_deslizante(imagen, kernel, progress_callback)

def realce_bordes(imagen, tam):
    mascara = -np.ones((tam, tam))
    mascara[tam//2, tam//2] = tam**2 - 1
    return mascara

def aplicar_realce_bordes(imagen, tam, umbral, progress_callback=None):

    kernel = realce_bordes(imagen, tam)

    respuesta = p.ventana_deslizante(
        imagen,
        kernel,
        progress_callback
    )

    return op.umbralizacion(
        respuesta,
        umbral
    )

def sobel():

    kernel_y = np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]
    ],dtype=float)

    kernel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ],dtype=float)

    return kernel_x, kernel_y

def aplicar_sobel(imagen, umbral, progress_callback=None):

    kernel_x, kernel_y = sobel()

    derivada_x = p.ventana_deslizante(
        imagen,
        kernel_x,
        progress_callback
    )

    derivada_y = p.ventana_deslizante(
        imagen,
        kernel_y,
        progress_callback
    )

    magnitud_gradiente = np.sqrt(
        derivada_x**2 +
        derivada_y**2
    )

    return p.normalizar(magnitud_gradiente)

def prewitt():

    kernel_y = np.array([
        [-1,-1,-1],
        [ 0, 0, 0],
        [ 1, 1, 1]
    ],dtype=float)

    kernel_x = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ],dtype=float)

    return kernel_x, kernel_y

def aplicar_prewitt(imagen, umbral, progress_callback=None):

    kernel_x, kernel_y = sobel()

    derivada_x = p.ventana_deslizante(
        imagen,
        kernel_x,
        progress_callback
    )

    derivada_y = p.ventana_deslizante(
        imagen,
        kernel_y,
        progress_callback
    )

    magnitud_gradiente = np.sqrt(
        derivada_x**2 +
        derivada_y**2
    )

    return p.normalizar(magnitud_gradiente)


    


