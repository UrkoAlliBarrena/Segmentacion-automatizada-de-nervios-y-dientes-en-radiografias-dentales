import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread
from skimage.morphology import area_closing
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import shutil
import stat

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

def postprocesado(ruta_etiquetas, ruta_imagenes, ruta_salida):
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
        for imagen in os.listdir(ruta_imagenes):
            nombre_imagen, _ = os.path.splitext(imagen)
            if nombre_archivo == nombre_imagen:
                ruta_imagen = os.path.join(ruta_imagenes, imagen)
                procesar_archivo(ruta_archivo, ruta_imagen, ruta_salida)

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

def calcular_metricas(imagen_predicha, imagen_original):
    '''
    Funcionalidad:
      Permite calcular las métricas de evaluación (precision, recall, f1, iou) entre la máscara del canal radicular
      predicho y la de la imagen original.

    Parámetros:
      - pred (numpy.ndarray): imagen de predicción después de aplicar area closing.
      - orig (numpy.ndarray): imagen original después de aplicar area closing.

    Returns:
      - precision, recall, f1, iou (tuple): métricas de evaluación.
    '''
    prediccion_aplanada = imagen_predicha.flatten()
    original_aplanada = imagen_original.flatten()
    precision = precision_score(original_aplanada, prediccion_aplanada, zero_division=1)
    recall = recall_score(original_aplanada, prediccion_aplanada, zero_division=1)
    f1 = f1_score(original_aplanada, prediccion_aplanada, zero_division=1)
    interseccion = np.sum(prediccion_aplanada & original_aplanada)
    union = np.sum(prediccion_aplanada | original_aplanada)
    iou = interseccion / union if union != 0 else 0
    return precision, recall, f1, iou

def procesar_par_imagenes(nombre_imagen, ruta_predichas, ruta_originales, ruta_mascaras_salida):
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
    ruta_predicciones = os.path.join(ruta_predichas, nombre_imagen)
    ruta_originales = os.path.join(ruta_originales, nombre_imagen)
    predicha = umbralizar(ruta_predicciones)
    original = umbralizar(ruta_originales)
    predicciones_closing = aplicar_area_closing(predicha)
    originales_closing = aplicar_area_closing(original)
    crear_directorio(ruta_mascaras_salida)
    nombre_imagen_a_guardar = os.path.join(ruta_mascaras_salida, nombre_imagen)
    imagen_a_guardar = (predicciones_closing.astype(np.uint8)) * 255
    cv2.imwrite(nombre_imagen_a_guardar, imagen_a_guardar)
    return calcular_metricas(predicciones_closing, originales_closing)

def obtener_metricas(ruta_predichas, ruta_originales, ruta_mascaras):
    '''
    Funcionalidad:
      Permite encontrar la imagen predicha en relación a la imagen original, para llevar a cabo el cálculo y obtención de
      las métricas de evaluación a partir de las máscaras generadas de los canales radiculares.

    Parámetros:
      - ruta_predichas (str): ruta de las imágenes predichas.
      - ruta_originales (str): ruta de las imágenes originales.

    Returns:
      - resumen, resumen_maximas, resumen_minimas (tuple): tupla con los diccionarios con la media de las métricas 
          obtenidas para todas las imágenes predichas, métricas máximas y mínimas obtenidas.
    '''
    imagenes_predicciones = os.listdir(ruta_predichas)
    precision_lista, recall_lista, iou_lista, f1_lista = [], [], [], []
    registro = []
    for nombre_imagen in imagenes_predicciones:
        metricas = procesar_par_imagenes(nombre_imagen, ruta_predichas, ruta_originales, ruta_mascaras)
        if metricas is not None:
            precision, recall, f1, iou = metricas
            precision_lista.append(precision)
            recall_lista.append(recall)
            iou_lista.append(iou)
            f1_lista.append(f1)
            registro.append({'ID': os.path.splitext(nombre_imagen)[0], 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'IoU': iou})
    exportar_metricas_excel(registro)
    resumen_maximas = {'Precision máxima': max(precision_lista), 'Recall máximo': max(recall_lista), 
               'IoU máximo': max(iou_lista), 'F1 máximo': max(f1_lista)}
    resumen_minimas = {'\nPrecision mínima': min(precision_lista), 'Recall mínimo': min(recall_lista), 
               'IoU mínimo': min(iou_lista), 'F1 mínimo': min(f1_lista)}
    resumen = {'\nPrecision media': np.array(precision_lista).mean(), 'Recall medio': np.array(recall_lista).mean(), 
               'IoU medio': np.array(iou_lista).mean(), 'F1 medio': np.array(f1_lista).mean()}
    registro.append({'ID': "Resumen métricas: máx, mín y media.", 'Precision': resumen_maximas, 'Recall': resumen_minimas, 'F1-Score': resumen, 'IoU': ""})
    return resumen, resumen_maximas, resumen_minimas

def exportar_metricas_excel(registros):
    """
    Funcionalidad:
        Permite crear un DataFrame a partir de las métricas almacenadas en un registro y se exporta como un archivo Excel.
        
    Parámetros:
      - registros (list): lista de diccionarios para cada caso, en el que se encuentra el nombre del caso y sus métricas.
        
    Returns:
        df (DataFrame): dataframe con las métricas para cada caso predicho.
    """
    df = pd.DataFrame.from_records(registros, columns=['ID', 'Precision', 'Recall', 'F1-Score', 'IoU'])
    df.to_excel('Metricas.xlsx', index=False)
    return df

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
            resultado = generar_imagen_resultado(imagen_binaria, imagen_skeleton)
            ruta_guardado = os.path.join(ruta_salida, nombre_imagen)
            cv2.imwrite(ruta_guardado, cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR))
    return imagen_skeleton

def eliminar_carpetas(carpetas):
    '''
    Funcionalidad:
      Permite eliminar las carpetas intermedias que se crean para la obtención de los resultados.

    Parámetros:
      - None: no recibe nada.

    Returns:
      - None: no devuelve nada.
    '''
    for carpeta in carpetas:
        if os.path.isdir(carpeta):
            shutil.rmtree(carpeta, onerror=lambda funcion, ruta, info: (os.chmod(ruta, stat.S_IWRITE), funcion(ruta)))

def binarizar_y_metricas(eliminar = True):
    """
    Funcionalidad:
        Permite ejecutar todo el proceso de obtención de métricas y el esqueleto de las máscaras predichas, además permite eliminar 
            carpetas intermedias (en caso de que no se quiera eliminar alguna se debe sacar de la lista "carpetas".

    Parámetros:
      - eliminar (bool): Parámetro que permite determinar si se desean eliminar las carpetas intermedias que se crean. Por defecto coge 
          el valor de True.i es True, elimina las carpetas intermedias tras completar el proceso.

    Returns:
      - None: no devuelve nada.
    """
    ruta_salida_predicciones = 'ROOT_predicciones' 
    ruta_salida_originales = 'ROOT_originales'

    ruta_etiquetas_predicciones = 'entrenamientos_y_predicciones/segment/predict/labels'
    ruta_etiquetas_originales = "Dataset/labels"

    ruta_imagenes = 'Dataset/images'

    print ("Obteniendo métricas...")
    postprocesado(ruta_etiquetas_originales, ruta_imagenes, ruta_salida_originales)
    postprocesado(ruta_etiquetas_predicciones, ruta_imagenes, ruta_salida_predicciones)
    ruta_salida_mascaras_predichas = "MASCARAS_PREDICHAS"
    metricas = obtener_metricas(ruta_salida_predicciones, ruta_salida_originales, ruta_salida_mascaras_predichas)
    print ("Métricas obtenidas")
    print (f"Resumen de métricas:\n{metricas}")
    print ("Obteniendo esqueleto de máscaras predichas...")
    
    ruta_salida = "MASCARAS_Y_SKELETON"
    obtener_imagenes_skeleton(ruta_salida_mascaras_predichas, ruta_salida)
    print ("Esqueleto obtenido")
    
    carpetas = ["ROOT_predicciones", "ROOT_originales", "MASCARAS_PREDICHAS", "MASCARAS_Y_SKELETON"]
    if eliminar == True:
        print ("Eliminando carpetas intermedias...")
        eliminar_carpetas(carpetas)
        print ("Carpetas intermedias eliminadas")
    print ("Finalizado")

if __name__ == "__main__":
    binarizar_y_metricas()