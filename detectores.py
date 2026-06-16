import numpy as np
import procesamiento as p


def _scale_progress(progress_callback, start, end):
    if progress_callback is None:
        return None

    def callback(pct):
        progress_callback(int(start + (end - start) * pct / 100))

    return callback


def _convolucion_raw(imagen, operador, progress_callback=None):
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


def canny(imagen, umbral_bajo=50, umbral_alto=100, progress_callback=None):
    """
    Detector de Canny simplificado.
    imagen: imagen en escala de grises uint8 (0-255)
    """

    # --------------------------
    # 1) Suavizado Gaussiano
    # --------------------------
    kernel_gauss = np.array([
        [2, 4, 5, 4, 2],
        [4, 9,12, 9, 4],
        [5,12,15,12, 5],
        [4, 9,12, 9, 4],
        [2, 4, 5, 4, 2]
    ], dtype=np.float32)

    kernel_gauss /= kernel_gauss.sum()

    suavizada = p.ventana_deslizante(
        imagen,
        kernel_gauss,
        progress_callback=_scale_progress(progress_callback, 0, 25)
    )

    # --------------------------
    # 2) Sobel
    # --------------------------
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx = _convolucion_raw(
        suavizada,
        kx,
        progress_callback=_scale_progress(progress_callback, 25, 45)
    )
    gy = _convolucion_raw(
        suavizada,
        ky,
        progress_callback=_scale_progress(progress_callback, 45, 65)
    )

    magnitud = np.sqrt(gx**2 + gy**2)
    direccion = np.rad2deg(np.arctan2(gy, gx))
    direccion[direccion < 0] += 180

    # Normalizar magnitud a 0-255
    if magnitud.max() > 0:
        magnitud = magnitud * 255.0 / magnitud.max()

    # --------------------------
    # 3) Supresión no máximos
    # --------------------------
    alto, ancho = magnitud.shape
    nms = np.zeros((alto, ancho), dtype=np.float32)

    pasos = max(1, alto // 50)
    for i in range(1, alto - 1):
        for j in range(1, ancho - 1):

            ang = direccion[i, j]

            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitud[i, j + 1]
                r = magnitud[i, j - 1]

            elif 22.5 <= ang < 67.5:
                q = magnitud[i + 1, j - 1]
                r = magnitud[i - 1, j + 1]

            elif 67.5 <= ang < 112.5:
                q = magnitud[i + 1, j]
                r = magnitud[i - 1, j]

            else:
                q = magnitud[i - 1, j - 1]
                r = magnitud[i + 1, j + 1]

            if magnitud[i, j] >= q and magnitud[i, j] >= r:
                nms[i, j] = magnitud[i, j]

        if progress_callback is not None and (i % pasos == 0 or i == alto - 2):
            _scale_progress(progress_callback, 45, 65)(int((i + 1) / alto * 100))

    # --------------------------
    # 4) Doble umbral
    # --------------------------
    fuerte = 255
    debil = 75

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    fuertes = nms >= umbral_alto
    debiles = (nms >= umbral_bajo) & (nms < umbral_alto)

    resultado[fuertes] = fuerte
    resultado[debiles] = debil

    # --------------------------
    # 5) Histéresis
    # --------------------------
    pasos = max(1, alto // 50)
    for i in range(1, alto - 1):
        for j in range(1, ancho - 1):

            if resultado[i, j] == debil:

                vecinos = resultado[i-1:i+2, j-1:j+2]

                if np.any(vecinos == fuerte):
                    resultado[i, j] = fuerte
                else:
                    resultado[i, j] = 0

        if progress_callback is not None and (i % pasos == 0 or i == alto - 2):
            _scale_progress(progress_callback, 65, 100)(int((i + 1) / alto * 100))

    return resultado


