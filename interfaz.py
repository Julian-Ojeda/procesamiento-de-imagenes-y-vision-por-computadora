import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

import procesamiento as p
import operadores as op
import ruidos as r
import filtros as f
import detectores as d
import sift as sf


# ---------------- VARIABLES ----------------

imagen_original = None
imagen_actual = None
selected_point = None
selection_mode = False


# ---------------- FUNCIONES ----------------

def cargar_imagen():

    global imagen_original, imagen_actual

    ruta = filedialog.askopenfilename(
        filetypes=[("Imagenes","*.jpg *.png *.bmp *.raw")]
    )

    if not ruta:
        return

    R,G,B = p.cargar_imagen(ruta)

    imagen_original = p.engrisar(R,G,B)
    imagen_actual = imagen_original.copy()

    mostrar_imagen(imagen_original,label_original)
    mostrar_imagen(imagen_actual,label_resultado)


def mostrar_imagen(imagen, label, highlight_point=None):

    imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    if imagen.ndim == 2:
        imagen_pintada = np.stack([imagen] * 3, axis=-1)
    else:
        imagen_pintada = imagen.copy()

    if highlight_point is not None:
        y, x = highlight_point
        alto_img, ancho_img = imagen_pintada.shape[:2]
        if 0 <= y < alto_img and 0 <= x < ancho_img:
            for dy in range(-2, 3):
                if 0 <= y + dy < alto_img:
                    imagen_pintada[y + dy, x] = [255, 255, 255]
            for dx in range(-2, 3):
                if 0 <= x + dx < ancho_img:
                    imagen_pintada[y, x + dx] = [255, 255, 255]

    pil = Image.fromarray(imagen_pintada)
    pil = pil.resize((360, 360))

    tk_img = ImageTk.PhotoImage(pil)

    label.config(image=tk_img)
    label.image = tk_img


def actualizar_progreso(valor):
    progress_var.set(valor)
    ventana.update_idletasks()


def reiniciar_progreso():
    progress_var.set(0)
    ventana.update_idletasks()


def seleccionar_punto(event):
    global selected_point, selection_mode
    if imagen_original is None or not selection_mode:
        return

    alto_img, ancho_img = imagen_original.shape
    label_w = event.widget.winfo_width()
    label_h = event.widget.winfo_height()
    x_img = int(event.x * ancho_img / max(label_w, 1))
    y_img = int(event.y * alto_img / max(label_h, 1))
    x_img = max(0, min(x_img, ancho_img - 1))
    y_img = max(0, min(y_img, alto_img - 1))

    selected_point = (y_img, x_img)
    selection_mode = False
    label_punto_seleccionado.config(text=f'Seleccionado: ({x_img}, {y_img})')
    label_original.config(cursor='')
    mostrar_imagen(imagen_original, label_original, highlight_point=selected_point)


def activar_seleccion(event=None):
    global selection_mode
    if imagen_original is None:
        return

    selection_mode = True
    label_original.config(cursor='crosshair')
    label_punto_seleccionado.config(text='Modo selección: haga clic en la imagen original')


def superponer_phi(imagen: np.ndarray, phi: np.ndarray) -> np.ndarray:
    imagen_rgb = np.stack([imagen] * 3, axis=-1).astype(np.uint8)
    mascara_lout = phi == 1
    mascara_lin = phi == -1
    imagen_rgb[mascara_lout] = np.array([255, 255, 0], dtype=np.uint8)
    imagen_rgb[mascara_lin] = np.array([255, 0, 0], dtype=np.uint8)
    return imagen_rgb


def aplicar_contorno_activo():
    global imagen_actual
    if imagen_original is None or selected_point is None:
        return

    rect_size = int(entry_tamanio_contorno.get())
    Na = int(entry_iteraciones_contorno.get())
    y_centro, x_centro = selected_point
    mitad = max(1, rect_size // 2)

    fila_inicio = max(0, y_centro - mitad)
    columna_inicio = max(0, x_centro - mitad)
    alto = min(rect_size, imagen_original.shape[0] - fila_inicio)
    ancho = min(rect_size, imagen_original.shape[1] - columna_inicio)

    rect_inicial = (fila_inicio, columna_inicio, alto, ancho)
    mascara_objeto, phi = d.aplicar_contornos_activos(imagen_original, rect_inicial, Na=Na, progress_callback=actualizar_progreso)

    imagen_actual = mascara_objeto
    overlay = superponer_phi(imagen_original, phi)
    mostrar_imagen(overlay, label_resultado)
    mostrar_imagen(imagen_original, label_original, highlight_point=selected_point)


def superponer_phi(imagen: np.ndarray, phi: np.ndarray) -> np.ndarray:
    imagen_rgb = np.stack([imagen] * 3, axis=-1).astype(np.uint8)
    mascara_lout = phi == 1
    mascara_lin = phi == -1
    imagen_rgb[mascara_lout] = np.array([255, 255, 0], dtype=np.uint8)
    imagen_rgb[mascara_lin] = np.array([255, 0, 0], dtype=np.uint8)
    return imagen_rgb


def aplicar_sift_keypoints():
    if imagen_original is None:
        return

    try:
        umbral_contraste = float(entry_sift_contraste.get())
    except ValueError:
        umbral_contraste = 5.0

    actualizar_progreso(0)
    try:
        keypoints_sift = sf.detectar_keypoints_sift(
            imagen_original,
            umbral_contraste=umbral_contraste,
            progress_callback=actualizar_progreso
        )
        label_sift_info.config(text=f'Keypoints: {len(keypoints_sift)}')
        overlay = sf.dibujar_keypoints(imagen_original, keypoints_sift)
        mostrar_imagen(overlay, label_resultado)
    finally:
        reiniciar_progreso()


def aplicar_sift_correspondencias():
    if imagen_original is None:
        return

    ruta = filedialog.askopenfilename(
        title='Elegir segunda imagen para comparar',
        filetypes=[("Imagenes", "*.jpg *.png *.bmp *.raw")]
    )
    if not ruta:
        return

    R, G, B = p.cargar_imagen(ruta)
    imagen_b = p.engrisar(R, G, B)

    try:
        umbral_contraste = float(entry_sift_contraste.get())
    except ValueError:
        umbral_contraste = 5.0
    try:
        umbral_distancia = float(entry_sift_distancia.get())
    except ValueError:
        umbral_distancia = 0.7

    actualizar_progreso(0)
    try:
        keypoints_a = sf.detectar_keypoints_sift(
            imagen_original, umbral_contraste=umbral_contraste, progress_callback=actualizar_progreso
        )
        keypoints_b = sf.detectar_keypoints_sift(imagen_b, umbral_contraste=umbral_contraste)
        correspondencias = sf.emparejar_descriptores(keypoints_a, keypoints_b, umbral_distancia=umbral_distancia)
        label_sift_info.config(text=f'Correspondencias: {len(correspondencias)}')

        viz = sf.dibujar_correspondencias(imagen_original, keypoints_a, imagen_b, keypoints_b, correspondencias)
        plt.figure()
        plt.imshow(viz)
        plt.axis('off')
        plt.title(f'{len(correspondencias)} correspondencias SIFT')
        plt.show()
    finally:
        reiniciar_progreso()


def aplicar_operador():
    umbral = int(entry_umbral.get())
    global imagen_actual

    if imagen_actual is None:
        return

    operacion = combo_operadores.get()

    if operacion == "Negativo":
        imagen_actual = op.negativo(imagen_actual)

    elif operacion == "Ecualizacion":
        imagen_actual = op.ecualizacion_histograma(imagen_actual)

    elif operacion == "Histograma":

        hist = op.histograma(imagen_actual)

        plt.figure()
        plt.plot(hist)
        plt.show()

        return
    elif operacion == "Umbralizacion":

        imagen_actual = op.umbralizacion(
            imagen_actual,
            umbral
    )

    mostrar_imagen(imagen_actual,label_resultado)



def aplicar_filtro():

    global imagen_actual

    if imagen_actual is None:
        return
    tam = int(combo_k.get())
    sigma = float(entry_sigma.get())
    lamda = float(entry_lambda.get())
    umbral = int(entry_umbral.get())
    ruido = combo_ruidos.get()  
    
    
    

    filtro = combo_filtros.get()

    actualizar_progreso(0)
    try:
        if filtro == "Gauss":
            imagen_actual = f.aplicar_filtro_gauss(
                imagen_actual,
                sigma,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Media":
            imagen_actual = f.aplicar_filtro_media(
                imagen_actual,
                tam,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Mediana":
            imagen_actual = f.filtro_mediana(
                imagen_actual,
                tam,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Mediana ponderada":
            imagen_actual = f.aplicar_filtro_mediana_ponderada(
                imagen_actual,
                tam,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Realce de bordes":

            umbral = int(entry_umbral.get())

            imagen_actual = f.aplicar_realce_bordes(
                imagen_actual,
                tam,
                umbral,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Definición":
            imagen_actual = f.aplicar_filtro_definicion(
                imagen_actual,
                tam,
                progress_callback=actualizar_progreso
            )

        elif filtro == "Sobel":

            imagen_actual = f.aplicar_sobel(
                imagen_actual,
                umbral,
                progress_callback=actualizar_progreso
            )


        elif filtro == "Prewitt":

            imagen_actual = f.aplicar_prewitt(
                imagen_actual,
                umbral,
                progress_callback=actualizar_progreso
            ) 

        mostrar_imagen(imagen_actual,label_resultado)
    finally:
        reiniciar_progreso()


def aplicar_detector_borde():
    global imagen_actual

    if imagen_actual is None:
        return

    umbral = int(entry_umbral.get())
    detector = combo_detectores.get()

    actualizar_progreso(0)
    try:
        if detector == 'Sobel':
            imagen_actual = f.aplicar_sobel(imagen_actual, umbral, progress_callback=actualizar_progreso)
        elif detector == 'Prewitt':
            imagen_actual = f.aplicar_prewitt(imagen_actual, umbral, progress_callback=actualizar_progreso)
        elif detector == 'Canny':
            imagen_actual = d.aplicar_filtro_canny(imagen_actual, umbral_bajo=umbral//2, umbral_alto=umbral)
        elif detector == 'SUSAN':
            umbral_similitud = max(1, min(umbral, 40))
            umbral_borde = min(8, max(1, umbral // 16))
            imagen_actual = d.aplicar_susan(
                imagen_actual,
                umbral_similitud=umbral_similitud,
                umbral_borde=umbral_borde,
                progress_callback=actualizar_progreso
            )
        elif detector == 'Hough':
            # Se asume que `imagen_actual` ya contiene la binaria de bordes (Canny)
            bordes = imagen_actual.copy()
            imagen_actual = d.aplicar_transformada_hough(
                imagen_original,
                imagen_bordes=bordes,
                pasos_theta=180,
                umbral_votos=umbral
            )

        mostrar_imagen(imagen_actual, label_resultado)
    finally:
        reiniciar_progreso()


def aplicar_ruido():
        

    global imagen_actual

    if imagen_actual is None:
        return

    tam = int(combo_k.get())
    sigma = float(entry_sigma.get())
    lamda = float(entry_lambda.get())
    umbral = int(entry_umbral.get())
    ruido = combo_ruidos.get()

    if ruido == "Sal y Pimienta":
        imagen_actual = r.ruido_sal_pimienta(
            imagen_actual,
            densidad = float(entry_densidad.get())
        )

    elif ruido == "Gaussiano":
        imagen_actual = r.agregar_ruido_gaussiano(
            imagen_actual,
            sigma
        )

    elif ruido == "Potencia":
        imagen_actual = r.transformacion_potencia(
            imagen_actual,
            lamda
        )

    mostrar_imagen(imagen_actual,label_resultado)
    reiniciar_progreso()


def restaurar():

    global imagen_actual

    if imagen_original is None:
        return

    imagen_actual = imagen_original.copy()

    mostrar_imagen(imagen_actual,label_resultado)


# ---------------- VENTANA ----------------

ventana = tk.Tk()
ventana.title("Procesamiento de Imagenes")
ventana.configure(bg="#1f2126")
ventana.state('zoomed')

style = ttk.Style(ventana)
style.theme_use('clam')
style.configure('TFrame', background='#1f2126')
style.configure('TLabel', background='#1f2126', foreground='#e6e6e6', font=('Helvetica', 10))
style.configure('Header.TLabel', background='#1f2126', foreground='#ff6b6b', font=('Helvetica', 16, 'bold'))
style.configure('Section.TLabel', background='#1f2126', foreground='#ff8a8a', font=('Helvetica', 11, 'bold'))
style.configure('Card.TLabelframe', background='#282b31', borderwidth=1, relief='solid')
style.configure('Card.TLabelframe.Label', background='#282b31', foreground='#ff6b6b', font=('Helvetica', 11, 'bold'))
style.configure('Accent.TButton', foreground='#ffffff', background='#a82a2a', padding=8)
style.map('Accent.TButton', background=[('active', '#cc2f2f')])
style.configure('Secondary.TButton', foreground='#ffb3b3', background='#2d3036', borderwidth=1, relief='solid', padding=8)
style.map('Secondary.TButton', background=[('active', '#3c4148')])
style.configure('TButton', foreground='#f0f0f0', background='#353940', padding=8)
style.configure('TCombobox', foreground='#f0f0f0', fieldbackground='#2d3036', background='#2d3036', padding=4)
style.configure('TLabelframe', background='#282b31')
style.configure('TLabelframe.Label', background='#282b31', foreground='#ff6b6b')

header_frame = ttk.Frame(ventana, padding=(20, 15))
header_frame.grid(row=0, column=0, sticky='ew')
ventana.columnconfigure(0, weight=1)

titulo = ttk.Label(header_frame, text='Procesamiento de Imágenes', style='Header.TLabel')
titulo.grid(row=0, column=0, sticky='w')

boton_frame = ttk.Frame(header_frame)
boton_frame.grid(row=0, column=1, sticky='e')

boton_cargar = ttk.Button(
    boton_frame,
    text='Cargar imagen',
    command=cargar_imagen,
    style='Accent.TButton'
)
boton_cargar.grid(row=0, column=0, padx=8)

boton_restaurar = ttk.Button(
    boton_frame,
    text='Restaurar original',
    command=restaurar,
    style='Secondary.TButton'
)
boton_restaurar.grid(row=0, column=1)

main_frame = ttk.Frame(ventana, padding=20)
main_frame.grid(row=1, column=0, sticky='nsew')
ventana.rowconfigure(1, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=0)

frame_imagenes = ttk.Frame(main_frame)
frame_imagenes.grid(row=0, column=0, sticky='nsew', padx=(0, 20), pady=(0, 10))
frame_imagenes.columnconfigure(0, weight=1)
frame_imagenes.columnconfigure(1, weight=1)

original_card = ttk.Labelframe(frame_imagenes, text='Original', style='Card.TLabelframe', padding=10)
original_card.grid(row=0, column=0, padx=8, pady=8, sticky='nsew')
result_card = ttk.Labelframe(frame_imagenes, text='Resultado', style='Card.TLabelframe', padding=10)
result_card.grid(row=0, column=1, padx=8, pady=8, sticky='nsew')

label_original = ttk.Label(original_card, text='Sin imagen', anchor='center', background='#282b31', foreground='#d8d8d8')
label_original.pack(expand=True, fill='both', ipadx=10, ipady=110)
label_original.bind('<Button-1>', seleccionar_punto)
label_resultado = ttk.Label(result_card, text='Sin imagen', anchor='center', background='#282b31', foreground='#d8d8d8')
label_resultado.pack(expand=True, fill='both', ipadx=10, ipady=110)

frame_controles = ttk.Labelframe(main_frame, text='Controles', style='Card.TLabelframe', padding=18)
frame_controles.grid(row=0, column=1, sticky='nsew')
frame_controles.columnconfigure(0, weight=1)
frame_controles.columnconfigure(1, weight=1)
frame_controles.columnconfigure(2, weight=1)
frame_controles.columnconfigure(3, weight=1)
frame_controles.columnconfigure(4, weight=1)

# OPERADORES

operadores_label = ttk.Label(frame_controles, text='Operadores', style='Section.TLabel')
operadores_label.grid(row=0, column=0, sticky='w', pady=(0, 6))

combo_operadores = ttk.Combobox(
    frame_controles,
    values=[
        'Negativo',
        'Ecualizacion',
        'Histograma',
        'Umbralizacion'
    ],
    state='readonly'
)
combo_operadores.grid(row=1, column=0, padx=8, pady=(0, 10), sticky='ew')
combo_operadores.current(0)

tt_button_operadores = ttk.Button(
    frame_controles,
    text='Aplicar',
    command=aplicar_operador
)
tt_button_operadores.grid(row=2, column=0, pady=4, sticky='ew')

# FILTROS

filtros_label = ttk.Label(frame_controles, text='Filtros', style='Section.TLabel')
filtros_label.grid(row=0, column=1, sticky='w', pady=(0, 6))

combo_filtros = ttk.Combobox(
    frame_controles,
    values=[
        'Gauss',
        'Media',
        'Mediana',
        'Mediana ponderada',
        'Realce de bordes',
        'Definición',
        'Sobel',
        'Prewitt'
    ],
    state='readonly'
)
combo_filtros.grid(row=1, column=1, padx=8, pady=(0, 10), sticky='ew')
combo_filtros.current(0)

boton_filtros = ttk.Button(
    frame_controles,
    text='Aplicar',
    command=aplicar_filtro
)
boton_filtros.grid(row=2, column=1, pady=4, sticky='ew')

# RUIDOS

ruidos_label = ttk.Label(frame_controles, text='Ruidos', style='Section.TLabel')
ruidos_label.grid(row=0, column=2, sticky='w', pady=(0, 6))

combo_ruidos = ttk.Combobox(
    frame_controles,
    values=[
        'Sal y Pimienta',
        'Gaussiano',
        'Potencia'
    ],
    state='readonly'
)
combo_ruidos.grid(row=1, column=2, padx=8, pady=(0, 10), sticky='ew')
combo_ruidos.current(0)

boton_ruidos = ttk.Button(
    frame_controles,
    text='Aplicar',
    command=aplicar_ruido
)
boton_ruidos.grid(row=2, column=2, pady=4, sticky='ew')

# DETECTORES DE BORDE

detector_label = ttk.Label(frame_controles, text='Detectores de borde', style='Section.TLabel')
detector_label.grid(row=0, column=3, sticky='w', pady=(0, 6))

combo_detectores = ttk.Combobox(
    frame_controles,
    values=[
        'Sobel',
        'Prewitt',
        'Canny',
        'SUSAN',
        'Hough'
    ],
    state='readonly'
)
combo_detectores.grid(row=1, column=3, padx=8, pady=(0, 10), sticky='ew')
combo_detectores.current(0)

boton_detectores = ttk.Button(
    frame_controles,
    text='Aplicar',
    command=aplicar_detector_borde
)
boton_detectores.grid(row=2, column=3, pady=4, sticky='ew')

# CONTORNO ACTIVO

contorno_frame = ttk.Labelframe(frame_controles, text='Contorno Activo', style='Card.TLabelframe', padding=10)
contorno_frame.grid(row=3, column=0, columnspan=5, sticky='ew', pady=(16, 0))
contorno_frame.columnconfigure(0, weight=1)
contorno_frame.columnconfigure(1, weight=1)

boton_seleccionar = ttk.Button(
    contorno_frame,
    text='Seleccionar punto',
    command=activar_seleccion,
    style='Secondary.TButton'
)
boton_seleccionar.grid(row=0, column=0, padx=8, pady=(0, 4), sticky='ew')

label_punto_seleccionado = ttk.Label(
    contorno_frame,
    text='Seleccionado: ninguno',
    background='#282b31',
    foreground='#f0f0f0'
)
label_punto_seleccionado.grid(row=0, column=1, padx=8, pady=(0, 4), sticky='ew')

rect_label = ttk.Label(contorno_frame, text='Tamaño rect.', style='TLabel')
rect_label.grid(row=1, column=0, sticky='w', padx=8)
entry_tamanio_contorno = ttk.Entry(contorno_frame)
entry_tamanio_contorno.grid(row=2, column=0, padx=8, pady=4, sticky='ew')
entry_tamanio_contorno.insert(0, '20')

iter_label = ttk.Label(contorno_frame, text='Iteraciones', style='TLabel')
iter_label.grid(row=1, column=1, sticky='w', padx=8)
entry_iteraciones_contorno = ttk.Entry(contorno_frame)
entry_iteraciones_contorno.grid(row=2, column=1, padx=8, pady=4, sticky='ew')
entry_iteraciones_contorno.insert(0, '150')

boton_contorno = ttk.Button(
    contorno_frame,
    text='Aplicar contorno',
    command=aplicar_contorno_activo,
    style='Accent.TButton'
)
boton_contorno.grid(row=3, column=0, columnspan=2, pady=4, sticky='ew')

# SIFT

sift_frame = ttk.Labelframe(frame_controles, text='SIFT', style='Card.TLabelframe', padding=10)
sift_frame.grid(row=4, column=0, columnspan=5, sticky='ew', pady=(16, 0))
sift_frame.columnconfigure(0, weight=1)
sift_frame.columnconfigure(1, weight=1)
sift_frame.columnconfigure(2, weight=1)
sift_frame.columnconfigure(3, weight=1)

contraste_sift_label = ttk.Label(sift_frame, text='Umbral contraste', style='TLabel')
contraste_sift_label.grid(row=0, column=0, sticky='w', padx=8)
entry_sift_contraste = ttk.Entry(sift_frame)
entry_sift_contraste.grid(row=1, column=0, padx=8, pady=(0, 4), sticky='ew')
entry_sift_contraste.insert(0, '5')

distancia_sift_label = ttk.Label(sift_frame, text='Distancia máx. match', style='TLabel')
distancia_sift_label.grid(row=0, column=1, sticky='w', padx=8)
entry_sift_distancia = ttk.Entry(sift_frame)
entry_sift_distancia.grid(row=1, column=1, padx=8, pady=(0, 4), sticky='ew')
entry_sift_distancia.insert(0, '0.7')

label_sift_info = ttk.Label(
    sift_frame,
    text='Keypoints: -',
    background='#282b31',
    foreground='#f0f0f0'
)
label_sift_info.grid(row=0, column=2, columnspan=2, padx=8, pady=(0, 4), sticky='ew')

boton_sift_keypoints = ttk.Button(
    sift_frame,
    text='Detectar keypoints',
    command=aplicar_sift_keypoints,
    style='Accent.TButton'
)
boton_sift_keypoints.grid(row=1, column=2, padx=8, pady=(0, 4), sticky='ew')

boton_sift_comparar = ttk.Button(
    sift_frame,
    text='Comparar con otra imagen',
    command=aplicar_sift_correspondencias,
    style='Secondary.TButton'
)
boton_sift_comparar.grid(row=1, column=3, padx=8, pady=(0, 4), sticky='ew')

# PARAMETROS

param_label = ttk.Label(frame_controles, text='Parámetros', style='Section.TLabel')
param_label.grid(row=8, column=0, columnspan=5, sticky='w', pady=(16, 6))

k_label = ttk.Label(frame_controles, text='K')
k_label.grid(row=9, column=0, sticky='w', padx=8)
combo_k = ttk.Combobox(
    frame_controles,
    values=[3, 5, 7, 9],
    state='readonly'
)
combo_k.grid(row=10, column=0, padx=8, pady=4, sticky='ew')
combo_k.current(0)

sigma_label = ttk.Label(frame_controles, text='Sigma')
sigma_label.grid(row=9, column=1, sticky='w', padx=8)
entry_sigma = ttk.Entry(frame_controles)
entry_sigma.grid(row=10, column=1, padx=8, pady=4, sticky='ew')
entry_sigma.insert(0, '1')

lambda_label = ttk.Label(frame_controles, text='Lambda')
lambda_label.grid(row=9, column=2, sticky='w', padx=8)
entry_lambda = ttk.Entry(frame_controles)
entry_lambda.grid(row=10, column=2, padx=8, pady=4, sticky='ew')
entry_lambda.insert(0, '1')

densidad_label = ttk.Label(frame_controles, text='Densidad')
densidad_label.grid(row=9, column=3, sticky='w', padx=8)
entry_densidad = ttk.Entry(frame_controles)
entry_densidad.grid(row=10, column=3, padx=8, pady=4, sticky='ew')
entry_densidad.insert(0, '0.05')

umbral_label = ttk.Label(frame_controles, text='Umbral')
umbral_label.grid(row=9, column=4, sticky='w', padx=8)
entry_umbral = ttk.Entry(frame_controles)
entry_umbral.grid(row=10, column=4, padx=8, pady=4, sticky='ew')
entry_umbral.insert(0, '128')

# Barra de progreso
progress_var = tk.IntVar(value=0)
progress_frame = ttk.Frame(frame_controles)
progress_frame.grid(row=6, column=0, columnspan=5, pady=(16, 0), sticky='ew')
progress_frame.columnconfigure(0, weight=0)
progress_frame.columnconfigure(1, weight=1)

progress_label = ttk.Label(progress_frame, text='Progreso:')
progress_label.grid(row=0, column=0, sticky='w')
progress_bar = ttk.Progressbar(
    progress_frame,
    variable=progress_var,
    maximum=100
)
progress_bar.grid(row=0, column=1, sticky='ew', padx=(8, 0))

# Ajuste final de tamaño
for i in range(7):
    frame_controles.rowconfigure(i, weight=0)

ventana.mainloop()