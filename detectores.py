import math

import numpy as np
import procesamiento as p


def _scale_progress(progress_callback, start, end):
    if progress_callback is None:
        return None

    def callback(pct):
        progress_callback(int(start + (end - start) * pct / 100))

    return callback


def convolucion(imagen, operador, progress_callback=None):
    alto_img, ancho_img = imagen.shape
    alto_k, ancho_k = operador.shape
    imagen_pad = p.padding(imagen, alto_k, ancho_k)
    resultado = np.zeros_like(imagen, dtype=np.float32)
    pasos = max(1, alto_img // 50)

    for y in range(alto_img):
        for x in range(ancho_img):
            ventana = imagen_pad[y : y + alto_k, x : x + ancho_k]
            resultado[y, x] = np.sum(ventana * operador)

        if progress_callback is not None and (y % pasos == 0 or y == alto_img - 1):
            progress_callback(int((y + 1) / alto_img * 100))

    return resultado



def aplicar_filtro_canny(imagen, umbral_bajo=30, umbral_alto=100):
    
    imagen_suavizada = aplicar_suavizado_gaussiano(imagen)
    
    magnitud_gradiente, angulo_gradiente = calcular_magnitud_y_angulo_gradiente(imagen_suavizada)
    
    bordes_finos = aplicar_supresion_no_maximos(magnitud_gradiente, angulo_gradiente)
    
    bordes_clasificados = doble_umbral(bordes_finos, umbral_bajo, umbral_alto)
    
    bordes_finales = conectar_bordes_por_histeresis(bordes_clasificados)
    
    return bordes_finales


def aplicar_suavizado_gaussiano(imagen):
    kernel_gaussiano = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
    ], dtype=np.float32)
    
    kernel_gaussiano /= kernel_gaussiano.sum()

    return p.ventana_deslizante(imagen, kernel_gaussiano)


def calcular_magnitud_y_angulo_gradiente(imagen_suavizada):
    operador_sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    operador_sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gradiente_x = convolucion(imagen_suavizada, operador_sobel_x)
    gradiente_y = convolucion(imagen_suavizada, operador_sobel_y)

    magnitud_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    
    if magnitud_gradiente.max() > 0:
        magnitud_gradiente = magnitud_gradiente * 255.0 / magnitud_gradiente.max()

    alto_imagen, ancho_imagen = magnitud_gradiente.shape
    angulo_gradiente = np.zeros((alto_imagen, ancho_imagen), dtype=np.float32)

    mascara_division_valida = gradiente_x != 0
    mascara_division_cero = gradiente_x == 0
# como te amo rad2deg, me solucionaste el problema de los ángulos negativos, gracias por existir
    angulo_gradiente[mascara_division_valida] = np.rad2deg(
        np.arctan(gradiente_y[mascara_division_valida] / gradiente_x[mascara_division_valida])
    )
    
    angulo_gradiente[mascara_division_cero] = 90.0
    angulo_gradiente[angulo_gradiente < 0] += 180.0

    return magnitud_gradiente, angulo_gradiente


def aplicar_supresion_no_maximos(magnitud_gradiente, angulo_gradiente):
    alto_imagen, ancho_imagen = magnitud_gradiente.shape
    supresion_no_maximos = np.zeros((alto_imagen, ancho_imagen), dtype=np.float32)

    for fila in range(1, alto_imagen - 1):
        for columna in range(1, ancho_imagen - 1):
            
            angulo_actual = angulo_gradiente[fila, columna]
            magnitud_actual = magnitud_gradiente[fila, columna]

            es_zona_amarilla = (0 <= angulo_actual < 22.5) or (157.5 <= angulo_actual <= 180)
            es_zona_verde = 22.5 <= angulo_actual < 67.5
            es_zona_azul = 67.5 <= angulo_actual < 112.5
            es_zona_roja = 112.5 <= angulo_actual < 157.5

            if es_zona_amarilla:
                vecino_1 = magnitud_gradiente[fila, columna + 1]
                vecino_2 = magnitud_gradiente[fila, columna - 1]
                
            elif es_zona_verde:
                vecino_1 = magnitud_gradiente[fila + 1, columna + 1]
                vecino_2 = magnitud_gradiente[fila - 1, columna - 1]
                
            elif es_zona_azul:
                vecino_1 = magnitud_gradiente[fila + 1, columna]
                vecino_2 = magnitud_gradiente[fila - 1, columna]
                
            elif es_zona_roja:
                vecino_1 = magnitud_gradiente[fila + 1, columna - 1]
                vecino_2 = magnitud_gradiente[fila - 1, columna + 1]
                
            else:
                vecino_1 = 0
                vecino_2 = 0

            es_maximo_local = (magnitud_actual >= vecino_1) and (magnitud_actual >= vecino_2)
            
            if es_maximo_local:
                supresion_no_maximos[fila, columna] = magnitud_actual

    return supresion_no_maximos


def doble_umbral(supresion_no_maximos, umbral_bajo, umbral_alto):
    alto_imagen, ancho_imagen = supresion_no_maximos.shape
    bordes_clasificados = np.zeros((alto_imagen, ancho_imagen), dtype=np.uint8)

    valor_borde_fuerte = 255
    valor_borde_debil = 75

    mascara_bordes_fuertes = supresion_no_maximos >= umbral_alto
    mascara_bordes_debiles = (supresion_no_maximos >= umbral_bajo) & (supresion_no_maximos < umbral_alto)

    bordes_clasificados[mascara_bordes_fuertes] = valor_borde_fuerte
    bordes_clasificados[mascara_bordes_debiles] = valor_borde_debil

    return bordes_clasificados


def conectar_bordes_por_histeresis(bordes_clasificados):
    alto_imagen, ancho_imagen = bordes_clasificados.shape
    bordes_finales = bordes_clasificados.copy()

    valor_borde_fuerte = 255
    valor_borde_debil = 75

    coordenadas_y, coordenadas_x = np.where(bordes_finales == valor_borde_fuerte)
    pixeles_fuertes_pendientes = list(zip(coordenadas_y, coordenadas_x))

    while pixeles_fuertes_pendientes:
        fila_actual, columna_actual = pixeles_fuertes_pendientes.pop()

        for desplazamiento_fila in [-1, 0, 1]:
            for desplazamiento_columna in [-1, 0, 1]:
                
                fila_vecino = fila_actual + desplazamiento_fila
                columna_vecino = columna_actual + desplazamiento_columna

                adentro_de_imagen = (0 <= fila_vecino < alto_imagen) and (0 <= columna_vecino < ancho_imagen)
                
                if adentro_de_imagen:
                    es_borde_debil = bordes_finales[fila_vecino, columna_vecino] == valor_borde_debil
                    
                    if es_borde_debil:
                        bordes_finales[fila_vecino, columna_vecino] = valor_borde_fuerte
                        pixeles_fuertes_pendientes.append((fila_vecino, columna_vecino))

    mascara_ruido_restante = bordes_finales == valor_borde_debil
    bordes_finales[mascara_ruido_restante] = 0

    return bordes_finales


def aplicar_susan(imagen, umbral_similitud=10, umbral_borde=2, progress_callback=None):
    """Detector SUSAN con ventana circular 27-conexa hardcodeada.

    La ventana es una máscara 7x7 con 27 píxeles alrededor del centro.
    Si pocos de esos píxeles tienen valor similar al centro, el punto es borde.
    """
    umbral_similitud = max(0, int(umbral_similitud))
    umbral_borde = min(max(int(umbral_borde), 1), 27)

    mascara_susan = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ], dtype=bool)

    imagen_int = imagen.astype(np.int16)
    alto_img, ancho_img = imagen_int.shape
    imagen_pad = p.padding(imagen_int, 7, 7)
    resultado = np.zeros((alto_img, ancho_img), dtype=np.float32)
    pasos = max(1, alto_img // 50)

    for y in range(alto_img):
        for x in range(ancho_img):
            centro = imagen_pad[y + 3, x + 3]
            vecindad = imagen_pad[y : y + 7, x : x + 7]
            similares = np.abs(vecindad - centro) <= umbral_similitud
            usansize = np.sum(similares & mascara_susan)

            # 27 = centro + 26 vecinos en la máscara 7x7.
            respuesta = max(0, 27 - usansize)
            resultado[y, x] = respuesta

        if progress_callback is not None and (y % pasos == 0 or y == alto_img - 1):
            progress_callback(int((y + 1) / alto_img * 100))

    bordes = (resultado >= umbral_borde).astype(np.uint8) * 255
    return bordes


def _esta_dentro_limites(alto: int, ancho: int, fila: int, columna: int) -> bool:
    return 0 <= fila < alto and 0 <= columna < ancho


def _vecinos_4(fila: int, columna: int) -> list[tuple[int, int]]:
    return [
        (fila - 1, columna),
        (fila + 1, columna),
        (fila, columna - 1),
        (fila, columna + 1),
    ]


def calcular_funcion_decision(imagen: np.ndarray, theta_0: float, theta_1: float) -> np.ndarray:
    """Calcula Fd(x) para cada píxel según el modelo de intercambio de píxeles."""
    eps = 1e-6
    imagen_float = imagen.astype(np.float32)
    distancia_fondo = np.abs(theta_0 - imagen_float) + eps
    distancia_objeto = np.abs(theta_1 - imagen_float) + eps
    return np.log(distancia_fondo / distancia_objeto)


def inicializar_contornos_activos(
    imagen: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]], float, float]:
    """Inicializa L_out, L_in y la matriz Phi a partir de un rectángulo inicial.

    rect = (fila_inicio, columna_inicio, alto, ancho)
    """
    alto_img, ancho_img = imagen.shape
    fila_inicio, columna_inicio, alto, ancho = rect
    fila_inicio = max(0, min(fila_inicio, alto_img - 1))
    columna_inicio = max(0, min(columna_inicio, ancho_img - 1))
    alto = max(1, min(alto, alto_img - fila_inicio))
    ancho = max(1, min(ancho, ancho_img - columna_inicio))

    phi = np.full((alto_img, ancho_img), 3, dtype=np.int8)
    L_in: set[tuple[int, int]] = set()
    L_out: set[tuple[int, int]] = set()

    interior = np.zeros((alto_img, ancho_img), dtype=bool)
    interior[fila_inicio:fila_inicio + alto, columna_inicio:columna_inicio + ancho] = True

    def tiene_vecino_exterior(fila: int, columna: int) -> bool:
        for vecino_fila, vecino_columna in _vecinos_4(fila, columna):
            if _esta_dentro_limites(alto_img, ancho_img, vecino_fila, vecino_columna):
                if not interior[vecino_fila, vecino_columna]:
                    return True
        return False

    for fila in range(fila_inicio, fila_inicio + alto):
        for columna in range(columna_inicio, columna_inicio + ancho):
            if tiene_vecino_exterior(fila, columna):
                L_in.add((fila, columna))
                phi[fila, columna] = -1
            else:
                phi[fila, columna] = -3

    for fila in range(fila_inicio - 1, fila_inicio + alto + 1):
        for columna in range(columna_inicio - 1, columna_inicio + ancho + 1):
            if not _esta_dentro_limites(alto_img, ancho_img, fila, columna):
                continue
            if interior[fila, columna]:
                continue
            for vecino_fila, vecino_columna in _vecinos_4(fila, columna):
                if _esta_dentro_limites(alto_img, ancho_img, vecino_fila, vecino_columna):
                    if interior[vecino_fila, vecino_columna]:
                        L_out.add((fila, columna))
                        phi[fila, columna] = 1
                        break

    valores_fondo = imagen[~interior]
    if valores_fondo.size == 0:
        theta_0 = float(imagen[fila_inicio, columna_inicio]) + 1.0
    else:
        theta_0 = float(valores_fondo.mean())

    theta_1 = float(imagen[fila_inicio:fila_inicio + alto, columna_inicio:columna_inicio + ancho].mean())

    return phi, L_out, L_in, theta_0, theta_1


def _es_interior(phi: np.ndarray, fila: int, columna: int) -> bool:
    return all(
        phi[vecino_fila, vecino_columna] in {-1, -3}
        for vecino_fila, vecino_columna in _vecinos_4(fila, columna)
        if _esta_dentro_limites(phi.shape[0], phi.shape[1], vecino_fila, vecino_columna)
    )


def _es_exterior(phi: np.ndarray, fila: int, columna: int) -> bool:
    return all(
        phi[vecino_fila, vecino_columna] in {1, 3}
        for vecino_fila, vecino_columna in _vecinos_4(fila, columna)
        if _esta_dentro_limites(phi.shape[0], phi.shape[1], vecino_fila, vecino_columna)
    )


def _expandir_L_out(
    imagen: np.ndarray,
    phi: np.ndarray,
    L_out: set[tuple[int, int]],
    L_in: set[tuple[int, int]],
    Fd: np.ndarray,
) -> None:
    for pixel in list(L_out):
        fila, columna = pixel
        if Fd[fila, columna] > 0:
            L_out.remove(pixel)
            L_in.add(pixel)
            phi[fila, columna] = -1
            for vecino_fila, vecino_columna in _vecinos_4(fila, columna):
                if _esta_dentro_limites(phi.shape[0], phi.shape[1], vecino_fila, vecino_columna):
                    if phi[vecino_fila, vecino_columna] == 3:
                        L_out.add((vecino_fila, vecino_columna))
                        phi[vecino_fila, vecino_columna] = 1


def _limpiar_L_in(phi: np.ndarray, L_in: set[tuple[int, int]]) -> None:
    for pixel in list(L_in):
        fila, columna = pixel
        if _es_interior(phi, fila, columna):
            L_in.remove(pixel)
            phi[fila, columna] = -3


def _contraer_L_in(
    phi: np.ndarray,
    L_in: set[tuple[int, int]],
    L_out: set[tuple[int, int]],
    Fd: np.ndarray,
) -> None:
    for pixel in list(L_in):
        fila, columna = pixel
        if Fd[fila, columna] < 0:
            L_in.remove(pixel)
            L_out.add(pixel)
            phi[fila, columna] = 1
            for vecino_fila, vecino_columna in _vecinos_4(fila, columna):
                if _esta_dentro_limites(phi.shape[0], phi.shape[1], vecino_fila, vecino_columna):
                    if phi[vecino_fila, vecino_columna] == -3:
                        L_in.add((vecino_fila, vecino_columna))
                        phi[vecino_fila, vecino_columna] = -1


def _limpiar_L_out(phi: np.ndarray, L_out: set[tuple[int, int]]) -> None:
    for pixel in list(L_out):
        fila, columna = pixel
        if _es_exterior(phi, fila, columna):
            L_out.remove(pixel)
            phi[fila, columna] = 3


def aplicar_contornos_activos(
    imagen: np.ndarray,
    rect_inicial: tuple[int, int, int, int],
    Na: int = 100,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    imagen_gris = imagen.astype(np.float32)
    phi, L_out, L_in, theta_0, theta_1 = inicializar_contornos_activos(imagen_gris, rect_inicial)
    Fd = calcular_funcion_decision(imagen_gris, theta_0, theta_1)

    for iteracion in range(1, Na + 1):
        # Paramos solo si NINGUNO de los dos frentes tiene más margen para moverse
        # (expansión Y contracción estancadas a la vez). Con "or" en vez de "and"
        # el algoritmo cortaba en la primera vuelta: los píxeles de L_in son los
        # que definieron theta_1, así que "Fd > 0 en todo L_in" es casi siempre
        # cierto desde el inicio y frenaba la expansión antes de que ocurriera.
        expansion_estancada = all(Fd[fila, columna] < 0 for fila, columna in L_out)
        contraccion_estancada = all(Fd[fila, columna] > 0 for fila, columna in L_in)
        if expansion_estancada and contraccion_estancada:
            break

        _expandir_L_out(imagen_gris, phi, L_out, L_in, Fd)
        _limpiar_L_in(phi, L_in)
        _contraer_L_in(phi, L_in, L_out, Fd)
        _limpiar_L_out(phi, L_out)

        if progress_callback is not None:
            progress_callback(int(iteracion / Na * 100))

    mascara_objeto = (phi < 0).astype(np.uint8) * 255
    return mascara_objeto, phi


# ------------------------------------------------------------------------------------------------------------------

def inicializar_tablero_votacion(imagen_binaria, pasos_theta):
    alto_imagen, ancho_imagen = imagen_binaria.shape
    
    theta_valores = np.linspace(-90, 90, pasos_theta)
    # aca hago trampa porque linespace me separa automaticamete todo en partes iguales, y no me deja tener el 90, entonces lo agrego manualmente
    diagonal_maxima = max(ancho_imagen, alto_imagen)
    r_maximo = math.sqrt(2) * diagonal_maxima
    
    pasos_r = int(2 * r_maximo)
    r_valores = np.linspace(-r_maximo, r_maximo, pasos_r)
    
    acumulador_vacio = np.zeros((pasos_r, pasos_theta), dtype=int)
    
    return acumulador_vacio, theta_valores, r_valores


def ejecutar_votacion_de_pixeles(imagen_binaria, acumulador_vacio, theta_valores, r_valores):
    coordenadas_y, coordenadas_x = np.where(imagen_binaria > 0)
    
    theta_radianes = np.deg2rad(theta_valores)
    acumulador_actualizado = acumulador_vacio.copy()

    for k in range(len(coordenadas_x)):
        x_k = coordenadas_x[k]
        y_k = coordenadas_y[k]
        
        for indice_theta, theta_actual_rad in enumerate(theta_radianes):
            
            rho_calculado = x_k * math.cos(theta_actual_rad) + y_k * math.sin(theta_actual_rad)
            
            indice_rho_cercano = np.argmin(np.abs(r_valores - rho_calculado))
            
            acumulador_actualizado[indice_rho_cercano, indice_theta] += 1
            
    return acumulador_actualizado



def dibujar_lineas_en_imagen(imagen, lineas, intensidad=255):
    imagen_lineas = np.zeros_like(imagen, dtype=np.uint8)
    alto_imagen, ancho_imagen = imagen.shape

    for rho, theta in lineas:
        theta_rad = math.radians(theta)
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)

        if abs(s) > 1e-6:
            for x in range(ancho_imagen):
                y = int(round((rho - x * c) / s))
                if 0 <= y < alto_imagen:
                    imagen_lineas[y, x] = intensidad
        else:
            for y in range(alto_imagen):
                x = int(round((rho - y * s) / c))
                if 0 <= x < ancho_imagen:
                    imagen_lineas[y, x] = intensidad

    return imagen_lineas


def construir_tablero_hough(bordes, pasos_theta=180):
    alto, ancho = bordes.shape
    # theta de -90 a 89 grados, onda simple y fácil de explicar
    thetas = np.deg2rad(np.linspace(-90, 90, pasos_theta, endpoint=False))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    max_dist = int(math.hypot(ancho, alto))
    rhos = np.arange(-max_dist, max_dist + 1)
    acumulador = np.zeros((len(rhos), len(thetas)), dtype=int)
    return acumulador, rhos, thetas, cos_t, sin_t, max_dist


def votar_bordes_en_hough(bordes, acumulador, cos_t, sin_t, offset):
    #encuentro las coordenadas de los pixeles de borde (donde bordes > 0)
    filas_borde, columnas_borde = np.nonzero(bordes)

    #para cada punto de borde, voto en el acumulador Hough para cada theta
    for x, y in zip(columnas_borde, filas_borde):
        for indice_theta, (cos_theta, sen_theta) in enumerate(zip(cos_t, sin_t)):
            # rho = x*cos(theta) + y*sin(theta)
            rho = int(round(x * cos_theta + y * sen_theta)) + offset
            acumulador[rho, indice_theta] += 1
# aca si o si tengo que pasarla por canny porque se rompe todo si no, porque el acumulador se llena de votos por ruido, y no se pueden extraer las líneas reales, entonces asumo que la imagen que me pasan ya es la binaria de bordes, y voto directamente sobre esa imagen

def extraer_lineas(acumulador, rhos, thetas, umbral_votos):
    indices = np.argwhere(acumulador >= umbral_votos)
    return [(rhos[r], np.rad2deg(thetas[t])) for r, t in indices]


def superponer_lineas(imagen, mask, rojo=False):
    if imagen.ndim == 2:
        base = np.stack([imagen] * 3, axis=-1).astype(np.uint8)
    else:
        base = imagen.copy().astype(np.uint8)
        if base.shape[2] == 4:
            base = base[:, :, :3]

    mask_indices = mask > 0
    if rojo:
        base[..., 0][mask_indices] = 255
        base[..., 1][mask_indices] = 0
        base[..., 2][mask_indices] = 0
    else:
        base[mask_indices] = 255
    return base


def aplicar_transformada_hough(imagen_original, imagen_bordes=None, pasos_theta=180, umbral_votos=100):
    # Si ya tenés la binaria de bordes, la uso directa.
    # Si no, tomo imagen_original como la binaria.
    if imagen_bordes is None:
        bordes = (imagen_original > 0).astype(np.uint8)
        imagen_base = None
    else:
        bordes = (imagen_bordes > 0).astype(np.uint8)
        imagen_base = imagen_original

    acumulador, rhos, thetas, cos_t, sin_t, max_dist = construir_tablero_hough(bordes, pasos_theta)
    votar_bordes_en_hough(bordes, acumulador, cos_t, sin_t, max_dist)
    lineas = extraer_lineas(acumulador, rhos, thetas, umbral_votos)

    if imagen_base is None:
        return dibujar_lineas_en_imagen(bordes, lineas)

    mask = dibujar_lineas_en_imagen(bordes, lineas, intensidad=1)
    return superponer_lineas(imagen_base, mask, rojo=True)

