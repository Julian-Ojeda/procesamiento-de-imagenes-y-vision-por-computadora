"""
Implementación de SIFT (Scale Invariant Feature Transform, Lowe 2004)
siguiendo la teórica de la cátedra (espacio-escala Gaussiano, DoG,
extremos 3D, histogramas de orientación y descriptor de 128 valores).

Simplificaciones respecto al paper original de Lowe (fuera del alcance
de la teórica): no se hace refinamiento sub-píxel de los keypoints, no
se descartan puntos sobre bordes con el test de curvatura de Hessiano,
y el histograma del descriptor no rota la grilla de muestreo (solo
rota el ángulo del gradiente contra la orientación del keypoint), por
lo que la invarianza a rotación es buena pero no perfecta.
"""

import numpy as np


# =====================================================================
# 1) ESPACIO ESCALA GAUSSIANO
# =====================================================================

def _kernel_gaussiano_1d(sigma: float) -> np.ndarray:
    """Kernel 1D de una Gaussiana G_sigma, con tamaño 8*sigma + 1 (según la
    teórica), redondeado al impar más cercano."""
    tam = int(round(8 * sigma))
    if tam % 2 == 0:
        tam += 1
    tam = max(tam, 3)

    radio = tam // 2
    xs = np.arange(-radio, radio + 1, dtype=np.float32)
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def _convolucion_1d(imagen: np.ndarray, kernel: np.ndarray, eje: int) -> np.ndarray:
    """Convoluciona 'imagen' con un kernel 1D a lo largo de un eje (0=filas,
    1=columnas). Como el Gaussiano es separable (G_sigma(x,y) = G_sigma(x) *
    G_sigma(y)), aplicar dos pasadas 1D da el mismo resultado que una
    convolución 2D completa pero mucho más rápido: el costo pasa de
    tam*tam multiplicaciones por píxel a 2*tam.

    En vez de recorrer píxel a píxel, para cada "tap" del kernel desplazamos
    toda la imagen y acumulamos: es la misma cuenta, vectorizada con numpy.
    """
    radio = len(kernel) // 2
    salida = np.zeros_like(imagen, dtype=np.float32)

    if eje == 0:
        pad = np.pad(imagen, ((radio, radio), (0, 0)), mode='edge')
        for i, peso in enumerate(kernel):
            salida += peso * pad[i:i + imagen.shape[0], :]
    else:
        pad = np.pad(imagen, ((0, 0), (radio, radio)), mode='edge')
        for i, peso in enumerate(kernel):
            salida += peso * pad[:, i:i + imagen.shape[1]]

    return salida


def aplicar_gauss(imagen: np.ndarray, sigma: float) -> np.ndarray:
    """Filtro Gaussiano G_sigma aplicado a toda la imagen (separable: primero
    filas, después columnas)."""
    kernel = _kernel_gaussiano_1d(sigma)
    borroneada = _convolucion_1d(imagen, kernel, eje=0)
    borroneada = _convolucion_1d(borroneada, kernel, eje=1)
    return borroneada


def submuestrear(imagen: np.ndarray) -> np.ndarray:
    """Reduce la resolución a la mitad promediando bloques de 2x2 píxeles
    (tal como lo describe la teórica: "reemplazando 4 pixels por el
    promedio entre ellos")."""
    alto, ancho = imagen.shape
    alto_par, ancho_par = alto - alto % 2, ancho - ancho % 2
    recorte = imagen[:alto_par, :ancho_par]
    # Truco de reshape: separamos cada eje en (mitad, 2) y promediamos sobre
    # los ejes de tamaño 2, que son los 2x2 vecinos de cada bloque.
    bloques = recorte.reshape(alto_par // 2, 2, ancho_par // 2, 2)
    return bloques.mean(axis=(1, 3)).astype(np.float32)


def construir_piramide_gaussiana(
    imagen: np.ndarray,
    num_octavas: int = 4,
    escalas_por_octava: int = 5,
    sigma0: float = 1.6,
) -> tuple[list[list[np.ndarray]], float]:
    """Construye la pirámide/espacio-escala Gaussiano: una lista de octavas,
    cada una con 'escalas_por_octava' imágenes cada vez más borroneadas.

    Dentro de una octava, el sigma absoluto de cada nivel es sigma0 * k^i.
    Elegimos k para que en 'escalas_por_octava' pasos el sigma se duplique
    (=una "octava" en el sentido musical que usa Lowe: el doble de sigma,
    así como una octava musical es el doble de frecuencia).

    Al pasar a la siguiente octava, se toma la imagen más borroneada de la
    octava actual y se reduce la resolución a la mitad (submuestreo).
    """
    k = 2 ** (1 / (escalas_por_octava - 1))

    piramide = []
    base = imagen.astype(np.float32)

    for _ in range(num_octavas):
        alto, ancho = base.shape
        if min(alto, ancho) < 16:
            break

        niveles = [aplicar_gauss(base, sigma0 * (k ** i)) for i in range(escalas_por_octava)]
        piramide.append(niveles)
        base = submuestrear(niveles[-1])

    return piramide, k


# =====================================================================
# 2) DIFERENCIAS DE GAUSSIANAS (DoG) Y KEYPOINTS
# =====================================================================

def construir_piramide_dog(piramide_gaussiana: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """D(x,y,sigma) = G(x,y,k*sigma) - G(x,y,sigma), para cada par de niveles
    consecutivos de cada octava."""
    piramide_dog = []
    for niveles in piramide_gaussiana:
        capas = [niveles[i + 1] - niveles[i] for i in range(len(niveles) - 1)]
        piramide_dog.append(capas)
    return piramide_dog


# Los 26 vecinos de un cubo 3x3x3 (3 escalas x 3x3 píxeles), sin el centro.
_VECINOS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
]


def encontrar_extremos_dog(capas_dog: list[np.ndarray], umbral_contraste: float) -> list[tuple[int, int, int]]:
    """Busca los keypoints candidatos: píxeles del DoG que son máximo o
    mínimo respecto a sus 26 vecinos (3x3x3 en espacio-escala), y que además
    superan un umbral de contraste (para descartar extremos "planos", poco
    confiables, típicos de zonas de bajo contraste o ruido).

    En vez de recorrer píxel por píxel (26 comparaciones cada uno, muy lento
    en Python puro), comparamos la imagen completa contra cada uno de sus 26
    vecinos desplazados con slicing de numpy: son 26 comparaciones
    vectorizadas sobre toda la imagen en vez de millones de comparaciones
    escalares.

    Devuelve una lista de (fila, columna, índice_de_escala_en_la_octava).
    """
    extremos = []

    for i in range(1, len(capas_dog) - 1):
        capas = {-1: capas_dog[i - 1], 0: capas_dog[i], 1: capas_dog[i + 1]}
        centro = capas[0][1:-1, 1:-1]
        alto, ancho = centro.shape

        es_maximo = np.ones((alto, ancho), dtype=bool)
        es_minimo = np.ones((alto, ancho), dtype=bool)

        for dz, dy, dx in _VECINOS_26:
            vecino = capas[dz][1 + dy:1 + dy + alto, 1 + dx:1 + dx + ancho]
            es_maximo &= centro >= vecino
            es_minimo &= centro <= vecino

        candidato = (es_maximo | es_minimo) & (np.abs(centro) > umbral_contraste)
        filas, columnas = np.nonzero(candidato)
        # +1 porque 'centro' es la imagen sin el borde de 1 píxel que se recortó arriba
        extremos.extend(zip(filas + 1, columnas + 1, [i] * len(filas)))

    return extremos


# =====================================================================
# 3) ASIGNACIÓN DE ORIENTACIONES
# =====================================================================

def _gradiente(imagen: np.ndarray, fila: int, columna: int) -> tuple[float, float]:
    """Magnitud y ángulo (en grados, 0-360) del gradiente en un píxel,
    estimados por diferencias finitas centrales con los vecinos."""
    gx = float(imagen[fila, columna + 1] - imagen[fila, columna - 1])
    gy = float(imagen[fila + 1, columna] - imagen[fila - 1, columna])
    magnitud = np.hypot(gx, gy)
    angulo = np.degrees(np.arctan2(gy, gx)) % 360
    return magnitud, angulo


def calcular_orientaciones_dominantes(
    imagen_octava: np.ndarray,
    fila: int,
    columna: int,
    sigma_local: float,
) -> list[float]:
    """Construye el histograma de orientaciones de 36 bins (10° cada uno) en
    una ventana de 16x16 alrededor del keypoint, ponderando cada gradiente
    por su módulo y por una Gaussiana de sigma = 1.5*escala (le da más peso
    a los píxeles cercanos al centro). La orientación dominante es el pico
    del histograma; si hay otros picos por encima del 80% del máximo, se
    generan keypoints adicionales con esas orientaciones (así un punto muy
    simétrico puede aportar varias orientaciones)."""
    radio = 8  # ventana de 16x16 => 8 píxeles para cada lado
    alto, ancho = imagen_octava.shape
    if not (radio <= fila < alto - radio and radio <= columna < ancho - radio):
        return []  # keypoint demasiado cerca del borde: no entra la ventana completa

    sigma_peso = 1.5 * sigma_local
    histograma = np.zeros(36, dtype=np.float32)

    for dy in range(-radio, radio):
        for dx in range(-radio, radio):
            magnitud, angulo = _gradiente(imagen_octava, fila + dy, columna + dx)
            peso_gaussiano = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma_peso ** 2))
            bin_idx = int(angulo // 10) % 36
            histograma[bin_idx] += magnitud * peso_gaussiano

    pico_maximo = histograma.max()
    if pico_maximo <= 0:
        return []

    orientaciones = []
    for b in range(36):
        if histograma[b] >= 0.8 * pico_maximo:
            orientaciones.append((b + 0.5) * 10.0)  # centro del bin, en grados
    return orientaciones


# =====================================================================
# 4) CONSTRUCCIÓN DEL DESCRIPTOR
# =====================================================================

def construir_descriptor(
    imagen_octava: np.ndarray,
    fila: int,
    columna: int,
    sigma_local: float,
    orientacion_kp: float,
) -> np.ndarray | None:
    """Descriptor de 128 = 8*16 valores: la ventana de 16x16 se parte en una
    grilla de 4x4 sub-regiones de 4x4 píxeles, y cada sub-región aporta un
    histograma de orientaciones de 8 bins (45° cada uno), ponderado por
    módulo del gradiente y por una Gaussiana de sigma = 0.5*escala.

    Para lograr invarianza a rotación, el ángulo de cada gradiente se mide
    relativo a la orientación del keypoint (orientacion_kp) antes de
    asignarlo a un bin. Al final se normaliza a norma 1, lo que le da al
    descriptor la invarianza a cambios de iluminación (una escena más clara
    u oscura escala todos los gradientes por igual, y la normalización
    cancela ese factor)."""
    radio = 8
    alto, ancho = imagen_octava.shape
    if not (radio <= fila < alto - radio and radio <= columna < ancho - radio):
        return None

    sigma_peso = 0.5 * sigma_local
    descriptor = np.zeros(128, dtype=np.float32)

    for sub_fila in range(4):
        for sub_columna in range(4):
            histograma_local = np.zeros(8, dtype=np.float32)
            for i in range(4):
                for j in range(4):
                    dy = -radio + sub_fila * 4 + i
                    dx = -radio + sub_columna * 4 + j
                    magnitud, angulo = _gradiente(imagen_octava, fila + dy, columna + dx)
                    angulo_relativo = (angulo - orientacion_kp) % 360
                    peso_gaussiano = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma_peso ** 2))
                    bin_idx = int(angulo_relativo // 45) % 8
                    histograma_local[bin_idx] += magnitud * peso_gaussiano

            indice_subregion = sub_fila * 4 + sub_columna
            descriptor[indice_subregion * 8:(indice_subregion + 1) * 8] = histograma_local

    norma = np.linalg.norm(descriptor)
    if norma > 1e-6:
        descriptor /= norma
    return descriptor


# =====================================================================
# 5) PIPELINE COMPLETO: DETECCIÓN DE KEYPOINTS SIFT
# =====================================================================

def detectar_keypoints_sift(
    imagen: np.ndarray,
    num_octavas: int = 4,
    escalas_por_octava: int = 5,
    sigma0: float = 1.6,
    umbral_contraste: float = 5.0,
    progress_callback=None,
) -> list[dict]:
    """Corre el pipeline completo de SIFT y devuelve una lista de keypoints,
    cada uno un diccionario con:
        x, y          -> posición en la imagen ORIGINAL (no en la octava)
        octava        -> octava donde se encontró
        escala        -> sigma efectivo, ya escalado a la imagen original
        orientacion   -> orientación dominante, en grados
        descriptor    -> vector numpy de 128 valores, normalizado
    """
    piramide, k = construir_piramide_gaussiana(imagen, num_octavas, escalas_por_octava, sigma0)
    piramide_dog = construir_piramide_dog(piramide)

    keypoints = []
    total_octavas = max(1, len(piramide_dog))

    for octava, capas_dog in enumerate(piramide_dog):
        extremos = encontrar_extremos_dog(capas_dog, umbral_contraste)
        imagenes_octava = piramide[octava]
        factor_a_original = 2 ** octava

        for fila, columna, indice_escala in extremos:
            sigma_local = sigma0 * (k ** indice_escala)
            imagen_octava = imagenes_octava[indice_escala]

            for orientacion in calcular_orientaciones_dominantes(imagen_octava, fila, columna, sigma_local):
                descriptor = construir_descriptor(imagen_octava, fila, columna, sigma_local, orientacion)
                if descriptor is None:
                    continue
                keypoints.append({
                    'x': columna * factor_a_original,
                    'y': fila * factor_a_original,
                    'octava': octava,
                    'escala': sigma_local * factor_a_original,
                    'orientacion': orientacion,
                    'descriptor': descriptor,
                })

        if progress_callback is not None:
            progress_callback(int((octava + 1) / total_octavas * 100))

    return keypoints


# =====================================================================
# 6) CÁLCULO DE CORRESPONDENCIAS ENTRE DOS IMÁGENES
# =====================================================================

def emparejar_descriptores(
    keypoints_a: list[dict],
    keypoints_b: list[dict],
    umbral_distancia: float = 0.7,
) -> list[tuple[int, int, float]]:
    """Para cada keypoint de A busca su vecino más cercano en B (distancia
    euclídea entre descriptores, ecuación de la teórica) y lo acepta como
    correspondencia si la distancia es menor al umbral. Como los
    descriptores están normalizados a norma 1, la distancia máxima posible
    entre dos de ellos es 2."""
    correspondencias = []

    for i, kp_a in enumerate(keypoints_a):
        mejor_distancia = np.inf
        mejor_j = -1
        for j, kp_b in enumerate(keypoints_b):
            distancia = float(np.linalg.norm(kp_a['descriptor'] - kp_b['descriptor']))
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_j = j
        if mejor_j >= 0 and mejor_distancia < umbral_distancia:
            correspondencias.append((i, mejor_j, mejor_distancia))

    return correspondencias


# =====================================================================
# 7) VISUALIZACIÓN
# =====================================================================

def _a_rgb(imagen: np.ndarray) -> np.ndarray:
    if imagen.ndim == 2:
        return np.stack([imagen] * 3, axis=-1).astype(np.uint8)
    return imagen[:, :, :3].astype(np.uint8)


def _dibujar_circulo(imagen: np.ndarray, cy: int, cx: int, radio: int, color, pasos: int = 24) -> None:
    alto, ancho = imagen.shape[:2]
    for i in range(pasos):
        angulo = 2 * np.pi * i / pasos
        y = int(round(cy + radio * np.sin(angulo)))
        x = int(round(cx + radio * np.cos(angulo)))
        if 0 <= y < alto and 0 <= x < ancho:
            imagen[y, x] = color


def _dibujar_linea(imagen: np.ndarray, y0: int, x0: int, y1: int, x1: int, color, pasos: int = 30) -> None:
    alto, ancho = imagen.shape[:2]
    for t in np.linspace(0, 1, pasos):
        y = int(round(y0 + (y1 - y0) * t))
        x = int(round(x0 + (x1 - x0) * t))
        if 0 <= y < alto and 0 <= x < ancho:
            imagen[y, x] = color


def dibujar_keypoints(imagen: np.ndarray, keypoints: list[dict], color=(255, 255, 0)) -> np.ndarray:
    """Dibuja cada keypoint como un círculo (radio proporcional a la escala)
    con una línea que indica su orientación dominante."""
    salida = _a_rgb(imagen)
    for kp in keypoints:
        x, y = int(round(kp['x'])), int(round(kp['y']))
        radio = max(2, int(round(kp['escala'] * 2)))
        _dibujar_circulo(salida, y, x, radio, color)
        angulo_rad = np.radians(kp['orientacion'])
        x2 = int(round(x + radio * np.cos(angulo_rad)))
        y2 = int(round(y + radio * np.sin(angulo_rad)))
        _dibujar_linea(salida, y, x, y2, x2, color)
    return salida


def dibujar_correspondencias(
    imagen_a: np.ndarray,
    keypoints_a: list[dict],
    imagen_b: np.ndarray,
    keypoints_b: list[dict],
    correspondencias: list[tuple[int, int, float]],
    color=(0, 255, 255),
) -> np.ndarray:
    """Arma un lienzo con las dos imágenes lado a lado y una línea por cada
    correspondencia encontrada, igual que los ejemplos de reconocimiento de
    objetos de la teórica."""
    rgb_a, rgb_b = _a_rgb(imagen_a), _a_rgb(imagen_b)
    alto = max(rgb_a.shape[0], rgb_b.shape[0])
    ancho_total = rgb_a.shape[1] + rgb_b.shape[1]

    lienzo = np.zeros((alto, ancho_total, 3), dtype=np.uint8)
    lienzo[:rgb_a.shape[0], :rgb_a.shape[1]] = rgb_a
    lienzo[:rgb_b.shape[0], rgb_a.shape[1]:] = rgb_b
    desplazamiento_x = rgb_a.shape[1]

    for idx_a, idx_b, _ in correspondencias:
        kp_a, kp_b = keypoints_a[idx_a], keypoints_b[idx_b]
        x0, y0 = int(kp_a['x']), int(kp_a['y'])
        x1, y1 = int(kp_b['x']) + desplazamiento_x, int(kp_b['y'])
        _dibujar_linea(lienzo, y0, x0, y1, x1, color, pasos=int(np.hypot(x1 - x0, y1 - y0)) + 1)

    return lienzo
