from ultralytics import YOLO
import os
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread
from skimage.morphology import area_closing
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import shutil
import stat
import math
import networkx as nw


def predecir_con_mejor_modelo(ruta_mejor_modelo, ruta_imagen):
    """
    Objetivo:
        Permite utilizar el mejor modelo entrenado para realizar las predicciones. Además, permite obtener las
        imágenes con los contornos predichos, y las coordenadas de dichos contornos.

    Parámetros:
        ruta_mejor_modelo (str): ruta del mejor modelo entrenado.
        directorio_datasets (str): ruta de los datasets con las imagenes de test.

    Returns:
        None: no devuelve nada
    """
    model = YOLO(ruta_mejor_modelo)
    predicciones = {}
    resultados = model.predict(ruta_imagen, save=True, save_txt=True)
    for resultado in resultados:
        nombre_archivo = os.path.basename(resultado.path)
        if hasattr(resultado, 'boxes') and resultado.boxes is not None:
            coordenadas = resultado.boxes.xyxy.tolist()
        elif hasattr(resultado, 'masks') and resultado.masks is not None:
            coordenadas = resultado.masks.xy.tolist()
        else:
            coordenadas = []
        predicciones[nombre_archivo] = coordenadas
    
def postprocesar_etiquetas(directorio_etiquetas = "runs/segment/predict/labels"):
    """
    Objetivo:
        Permite postprocesar las etiquetas obtenidas de las predicciones realizadas para quedarse sólo con la información más relevante,
            en este caso, el contorno que tenga mayor longitud para cada clase.

    Parámetros:
        directorio_etiquetas (str): ruta de los archivos que contienen las etiquetas en formato (.txt), Por defecto adquiere el valor de
                                    "/content/runs/segment/predict/labels".

    Returns:
        resultados (dict): diccionario con sólo un conjunto de coordenadas para cada clase.
    """
    resultados = {}
    for archivo in os.listdir(directorio_etiquetas):
        if archivo.endswith('.txt'):
            ruta_archivo = os.path.join(directorio_etiquetas, archivo)
            with open(ruta_archivo, 'r') as fichero:
                lineas = fichero.readlines()
            clases = defaultdict(lambda: (None, -1))
            for linea in lineas:
                partes = linea.strip().split()
                clase = partes[0]
                coordenadas = partes[1:]
                cantidad_coordenadas = len(coordenadas) // 2
                if cantidad_coordenadas > clases[clase][1]:
                    clases[clase] = (linea.strip(), cantidad_coordenadas)
            resultados[archivo] = [coordenadas for coordenadas, cantidad in clases.values()]
    return resultados

def guardar_predicciones_postprocesadas(directorio_salida="predicciones_postprocesadas"):
    """
    Objetivo:
        Permite almacenar las predicciones postprocesadas en archivos con formato (.txt).

    Parámetros:
        directorio_salida (str): ruta para almacenar las coordenadas en formato (.txt).

    Returns:
        None: no devuelve nada.
    """
    os.makedirs(directorio_salida, exist_ok=True)
    resultados = postprocesar_etiquetas()
    for nombre_archivo, lista_coordenadas in resultados.items():
        ruta_salida = os.path.join(directorio_salida, nombre_archivo)
        with open(ruta_salida, 'w') as fichero_texto:
            for coordenadas in lista_coordenadas:
                fichero_texto.write(coordenadas + '\n')   

def crear_directorio(ruta):
    '''
    Funcionalidad:
      Permite crear un nuevo directorio si este no existe.
    
    Parámetros:
      - ruta (str): ruta donde se va a crear el nuevo directorio.
    
    Returns:
      - None: no devuelve nada.
    '''
    os.makedirs(ruta, exist_ok=True)

def cargar_imagen(ruta_imagen):
    '''
    Funcionalidad:
      Permite cargar una imagen, convertirla en una imagen a color RGB, y se obtienen las dimensiones de dicha imagen.
    
    Parámetros:
      - ruta_imagen (str): ruta de la imagen.
    
    Returns:
      - imagen, ancho, alto (tuple): tupla con la imagen en RGB, y sus dimensiones.
    '''
    imagen = Image.open(ruta_imagen).convert("RGB")
    ancho, alto = imagen.size
    return imagen, ancho, alto

def procesar_etiquetas(ruta_etiqueta, ancho, alto):
    '''
    Funcionalidad:
      Permite procesar un archivo de etiquetas para dibujar las coordenadas presentes en dicho archivo, con el objetivo de
      dibujar el contorno cerrado que describe el canal radicular de un diente. Este contorno se dibuja únicamente si en el
      archivo de etiquetas está la clase 0 presente.
    
    Parámetros:
      - ruta_etiqueta (str): ruta del archivo que contiene las etiquetas (coordenadas).
      - ancho (int): ancho necesario para la imagen de salida.
      - alto (int): alto dnecesario para la imagen de salida.
    
    Returns:
      - tiene_clase0, imagen_resultado (tuple): como primer elemento un booleano que indica si el archivo de etiquetas 
            presenta una clase 0 o no. Y en segundo lugar la imagen con el contorno del canal radicular dibujado.
    '''
    tiene_clase0 = False
    imagen_resultado = np.zeros((alto, ancho), dtype=np.uint8)
    with open(ruta_etiqueta, 'r') as fichero:
        lineas = fichero.readlines()
    for linea in lineas:
        partes = linea.strip().split()
        clase = partes[0]
        coordenadas = partes[1:]
        coordenadas_sin_normalizar = []
        for i in range(0, len(coordenadas), 2):
            x = float(coordenadas[i]) * ancho
            y = float(coordenadas[i + 1]) * alto
            coordenadas_sin_normalizar.append([x, y])
        puntos = np.array(coordenadas_sin_normalizar, dtype=np.int32).reshape((-1, 1, 2))
        if clase == '0':
            tiene_clase0 = True
            cv2.polylines(imagen_resultado, [puntos], isClosed=True, color=(255), thickness=1)
    return tiene_clase0, imagen_resultado

def procesar_archivo(ruta_etiqueta, ruta_imagen, ruta_salida):
    '''
    Funcionalidad:
      Permite cargar una imagen para obtener sus dimensiones, y seguidamente procesar el archivo que contiene las etiquetas
      o coordenadas, para dibujar el contorno asociado al canal radicular. En caso de que exista la clase asociada a dicho
      contorno, se almacena la imagen generada en el directorio de salida.
    
    Parámetros:
      - ruta_etiqueta (str): ruta del archivo que contiene las etiquetas (coordenadas).
      - ruta_imagen (str): ruta de la imagen original.
      - ruta_salida (str): ruta del directorio de salida donde se guardarán las imagenes procesadas.
    
    Returns:
      - None: no devuelve nada.
    '''
    imagen_cargada, ancho, alto = cargar_imagen(ruta_imagen)
    tiene_clase0, imagen_procesada = procesar_etiquetas(ruta_etiqueta, ancho, alto)
    if tiene_clase0:
        nombre_base = os.path.splitext(os.path.basename(ruta_etiqueta))[0]
        ruta_destino = os.path.join(ruta_salida, f"{nombre_base}.png")
        cv2.imwrite(ruta_destino, imagen_procesada)

def postprocesado(ruta_etiquetas, ruta_imagen, ruta_salida):
    '''
    Funcionalidad:
      Permite crear el directorio de salida donde se almacenarán los contornos dibujados. Para ello recorre los archivos de
      etiquetas e imágenes, y en caso de que coincidan en nombre se emparejan para su procesamiento.
    
    Parámetros:
      - ruta_etiqueta (str): ruta de los archivos que contienen las etiquetas (coordenadas).
      - ruta_imagen (str): ruta de las imágenes originales.
      - ruta_salida (str): ruta del directorio de salida donde se guardarán las imagenes procesadas.
    
    Returns:
      - None: no devuelve nada.
    '''
    crear_directorio(ruta_salida)
    for archivo in os.listdir(ruta_etiquetas):
        nombre_archivo, _ = os.path.splitext(archivo)
        ruta_archivo = os.path.join(ruta_etiquetas, archivo)
        for imagen in os.listdir(ruta_imagen):
            nombre_imagen, _ = os.path.splitext(imagen)
            if nombre_archivo == nombre_imagen:
                ruta_imagen_completa = os.path.join(ruta_imagen, imagen)
                procesar_archivo(ruta_archivo, ruta_imagen_completa, ruta_salida)

def umbralizar(ruta_imagen):
    '''
    Funcionalidad:
      Permite umbralizar imágenes RGB. Las imágenes que se van a umbralizar proceden de las obtenidas mediante la 
      función _postprocesado()_

    Parámetros:
      - ruta_imagen (str): ruta de la imagen que se quiere umbralizar.

    Returns:
      - imagen_bin (numpy.ndarray): imagen umbralizada (binarizada).
    '''
    imagen = imread(ruta_imagen)
    imagen_binarizada = imagen > 254
    return imagen_binarizada

def aplicar_area_closing(imagen):
    '''
    Funcionalidad:
      Permite aplicar el operador de cierre sobre una imagen binarizada, con el objetivo de obtener la máscara que limita
      el contorno de la imagen binaria. Como umbral se utiliza el área del producto de la mitad del ancho por la mitad del 
      alto de la imagen.

    Parámetros:
      - imagen (numpy.ndarray): imagen binarizada sobre la que se va a aplicar el area closing o cierre de área.

    Returns:
      - imagen_cerrada (numpy.ndarray): imagen resultante del cierre.
    '''
    ancho, alto = imagen.shape
    area_threshold = (ancho // 2) * (alto // 2)
    imagen_cerrada = area_closing(imagen, area_threshold=area_threshold)
    return imagen_cerrada

def procesar_imagen(ruta_imagen, ruta_mascaras_salida):
    '''
    Funcionalidad:
      Permite obtener dos imágenes (original y predicha), umbralizarlas, aplicar el cierre de área, almacenarlas en una
      nueva carpeta y calcular las métricas.
      de evaluación entre las máscaras de ambas imágenes.

    Parámetros:
      - nombre_imagen (str): nmbre de la imagen a procesar.
      - ruta_predichas (str): ruta de las imágenes predichas.
      - ruta_originales (str): ruta de las imágenes originales.

    Returns:
      - calcular_metricas(pred_closing, orig_closing) (tuple): métricas de evaluación.
    '''
    predicha = umbralizar(ruta_imagen)
    predicciones_closing = aplicar_area_closing(predicha)
    crear_directorio(ruta_mascaras_salida)
    nombre_imagen_a_guardar = os.path.join(ruta_mascaras_salida, "CASO_1.png")
    imagen_a_guardar = (predicciones_closing.astype(np.uint8)) * 255
    cv2.imwrite(nombre_imagen_a_guardar, imagen_a_guardar)      
    return imagen_a_guardar

def binarizar_imagen(ruta_imagen):
    '''
    Funcionalidad:
      Permite binarizar una imagen, en función del tipo de imagen que se tenga en la ruta especificada.

    Parámetros:
      - ruta_imagen (str): ruta de la imagen que se quiere binarizar.

    Returns:
      - imagen_binaria (ndarray): imagen binarizada.
    '''
    imagen_original = imread(ruta_imagen)
    if len(imagen_original.shape) == 3 and imagen_original.shape[2] == 3:
        imagen_gris = rgb2gray(imagen_original)
    else:
        imagen_gris = imagen_original
    imagen_binaria = imagen_gris > 0.5
    return imagen_binaria

def generar_imagen_resultado(imagen_binaria, imagen_skeleton):
    '''
    Funcionalidad:
      Permite obtener una imagen resultado de la combinación de la máscara y el esqueleto de la misma.

    Parámetros:
      - imagen_binaria (ndarray): imagen binaria que representa la máscara del canal radicular.
      - imagen_skeleton (ndarray): imagen binaria del esqueleto.

    Returns:
      - resultado (ndarray): imagen RGB con la máscara en azul y el esqueleto en blanco.
    '''
    resultado = np.zeros((*imagen_binaria.shape, 3), dtype=np.uint8)
    resultado[imagen_binaria] = [0, 0, 255]
    resultado[imagen_skeleton] = [255, 255, 255]
    return resultado

def obtener_imagenes_skeleton(ruta_entrada, ruta_salida):
    '''
    Funcionalidad:
      Permite obtener un directorio en el que se almacenan las máscaras de los canales radiculares y el esqueleto de las
      máscaras.

    Parámetros:
      - ruta_entrada (str): ruta de las máscaras predichas.
      - ruta_salida (str): ruta de las imágenes con esqueleto.

    Returns:
      - None: no devuelve nada.
    '''
    crear_directorio(ruta_salida)
    for nombre_imagen in os.listdir(ruta_entrada):
        if nombre_imagen.lower().endswith(('.png')):
            ruta_imagen = os.path.join(ruta_entrada, nombre_imagen)
            imagen_binaria = binarizar_imagen(ruta_imagen)
            imagen_skeleton = skeletonize(imagen_binaria)
            esqueleto_podado = extraer_tronco_principal((imagen_skeleton > 0).astype(np.uint8))
            resultado = generar_imagen_resultado(imagen_binaria, esqueleto_podado)
            ruta_guardado = os.path.join(ruta_salida, nombre_imagen)
            cv2.imwrite(ruta_guardado, cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR))
    return esqueleto_podado

def obtener_longitud_esqueleto(imagen, anchura = 0.0258, altura = 0.0256):
    '''
    Funcionalidad:
      Permite calcular la longitud del esqueleto de una imagen binaria de segmentación de canal radicular. La anchura y altura toman los valores por defecto
      establecidos por el odontólogo.

    Parámetros:
      - imagen (string): ruta de la imagen (escala de grises) donde el esqueleto está representado con píxeles de color blanco; es decir, 1.
      - anchura (float): escala en milímetros por píxel en el eje horizontal. Por defecto 0.0258 milímetros.
      - altura (float): escala en milímetros por píxel en el eje vertical. Por defecto 0.0256 milímetros

    Returns:
      - longitud (float): longitud total del esqueleto en milímetros.
    '''
    diagonal = math.sqrt(anchura**2 + altura**2)
    if imagen.ndim == 3:
        imagen = imagen[..., 0]
    esqueleto = (imagen > 0).astype(np.uint8)
    coordenadas_y, coordenadas_x = np.where(esqueleto == 1)
    coordenadas = list(zip(coordenadas_y, coordenadas_x))
    longitud = 0
    for indice_coordenada in range(1, len(coordenadas)):
        coordenada_y_anterior, coordenada_x_anterior = coordenadas[indice_coordenada-1]
        coordenada_y_actual, coordenada_x_actual = coordenadas[indice_coordenada]
        distancia_y, distancia_x = abs(coordenada_y_actual - coordenada_y_anterior), abs(coordenada_x_actual - coordenada_x_anterior)
        if distancia_x == 1 and distancia_y == 1:
            longitud += diagonal
        elif distancia_x == 1 and distancia_y == 0:
            longitud += anchura
        elif distancia_x == 0 and distancia_y == 1:
            longitud += altura
    return f"Longitud: {longitud:.4f} mm"


def construir_grafo(esqueleto):
    """
    Objetivo:
        Permite construir un grafo no dirigido a partir de una imagen de esqueleto binaria, mediante la distancia euclídea entre nodos.

    Parámetros:
        esqueleto (numpy.ndarray): Imagen binaria del esqueleto.

    Returns:
        grafo (networkx.Graph): Grafo que representa el esqueleto podado.
    """
    grafo = nw.Graph()
    altura, anchura = esqueleto.shape
    vecinos = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(altura):
        for x in range(anchura):
            if esqueleto[y,x]:
                grafo.add_node((y,x))
                for desplazamiento_y,desplazamiento_x in vecinos:
                    coordenada_y_vecino, coordenada_x_vecino = y+desplazamiento_y, x+desplazamiento_x
                    if 0 <= coordenada_y_vecino < altura and 0 <= coordenada_x_vecino < anchura and esqueleto[coordenada_y_vecino, coordenada_x_vecino]:
                        distancia_euclidea = math.hypot(desplazamiento_y,desplazamiento_x)
                        grafo.add_edge((y,x), (coordenada_y_vecino, coordenada_x_vecino), weight=distancia_euclidea)
    return grafo

def encontrar_endpoints(grafo):
    """
    Objetivo:
        Permite encontrar los nodos del grafo que tienen solo un vecino.

    Parámetros:
        grafo (networkx.Graph): Grafo que representa el esqueleto podado.

    Returns:
        list: Lista de nodos con grado 1.
    """
    endpoints = []
    for nodo, conexion in grafo.degree():
        if conexion == 1:
            endpoints.append(nodo)
    return endpoints

def extraer_tronco_principal(esqueleto):
    """
    Objetivo:
        Permite obtener la rama más larga del esqueleto y eliminar las ramificaciones.

    Parámetros:
        esqueleto (numpy.ndarray): Imagen binaria del esqueleto.

    Returns:
        esqueleto_podado (numpy.ndarray): Imagen binaria que representa el tronco principal del esqueleto. Es decir, el esqueleto podado sin
            sus ramificaciones.
    """
    grafo = construir_grafo(esqueleto)
    endpoints = encontrar_endpoints(grafo)
    longitud_maxima = 0
    mejor_camino = None
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            endpoint_1, endpoint_2 = endpoints[i], endpoints[j]
            try: 
                camino = nw.shortest_path_length(grafo, endpoint_1, endpoint_2, weight='weight')
                if camino > longitud_maxima:
                    longitud_maxima = camino
                    mejor_camino = nw.shortest_path(grafo, endpoint_1, endpoint_2, weight='weight')
            except:
                continue
    esqueleto_podado = np.zeros_like(esqueleto, dtype=np.uint8)
    for (y,x) in mejor_camino:
        esqueleto_podado[y,x] = 1
    return esqueleto_podado

def eliminar_carpetas():
    '''
    Funcionalidad:
      Permite eliminar las carpetas intermedias que se crean para la obtención de los resultados.

    Parámetros:
      - None: no recibe nada.

    Returns:
      - None: no devuelve nada.
    '''
    carpetas = ["predicciones_postprocesadas", "runs", "FINAL", "MASCARAS_FINAL", "SKELETON"]
    for carpeta in carpetas:
        if os.path.isdir(carpeta):
            shutil.rmtree(carpeta, onerror=lambda funcion, ruta, info: (os.chmod(ruta, stat.S_IWRITE), funcion(ruta)))