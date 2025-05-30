!nvidia-smi
!pip install ultralytics roboflow opencv-python pandas

import os
import zipfile
import pandas as pd
import shutil
from google.colab import drive, files
from ultralytics import YOLO
import json
from collections import defaultdict

def montar_drive():
    """
    Objetivo:
        Permite cargar la cuenta de Google Drive en el entorno de ejecución de Google Colab.

    Parámetros:
        None: no devuelve nada.

    Returns:
        None: no devuelve nada.
    """
    drive.mount('/content/drive')

def extraer_datasets(ruta_archivo_zip, directorio_salida):
    """
    Objetivo:
        Permite extraer el contenido de un archivo ZIP que almacena los datasets en un directorio específico.

    Parámetros:
        ruta_zip (str): ruta del archivo ZIP que se quiere extraer.
        directorio_salida (str): directorio donde se almacenarán los datasets una vez descomprimida la carpeta que los almacena.

    Returns:
        None: no devuelve nada.
    """
    with zipfile.ZipFile(ruta_archivo_zip, 'r') as archivo_zip:
        archivo_zip.extractall(directorio_salida)

def entrenamiento_validacion(directorio_datasets, indice_dataset):
    """
    Objetivo:
        Permite realizar el entrenamiento y la validación del modelo YOLO11 para segmentar las imágenes almacenadas en los datasets extraídos previamente.

    Parámetros:
        directorio_datasets (str): ruta del directorio donde se encuentran los datasets almacenados.
        indice_dataset (int): índice del dataset que se va a utilizar para realizar el entrenamiento.

    Returns:
        None: no devuelve nada.
    """
    ruta_data_yaml = os.path.join(directorio_datasets, f'dataset{indice_dataset}.yaml')
    model = YOLO('yolo11n-seg.pt')
    model.train(task='segment', mode='train', data=ruta_data_yaml, epochs=300, val=True, patience=100, augment=True, imgsz=1024)#, save_json=True)

def obtener_datasets(directorio_datasets):
    """
    Objetivo:
        Permite obtener, y a la vez ordenar las rutas de los datasets en un directorio.

    Parámetros:
        directorio_datasets (str): ruta del directorio donde se almacenan los datasets.

    Returns:
        list: lista con las rutas de los datasets ordenados según su índice.
    """
    return sorted([os.path.join(directorio_datasets, i) for i in os.listdir(directorio_datasets) if os.path.isdir(os.path.join(directorio_datasets, i))],
        key=lambda x: int(x.split('Dataset')[-1]))

def ejecutar_entrenamiento(directorio_datasets):
    """
    Objetivo:
        Permite ejecutar el proceso completo de entrenamiento y validación para cada dataset.

    Parámetros:
        directorio_datasets (str): ruta del directorio donde se almacenan los datasets.

    Returns:
        None: no devuelve nada.
    """
    datasets = obtener_datasets(directorio_datasets)
    for indice_dataset, dataset in enumerate(datasets, start=1):
        entrenamiento_validacion(dataset, indice_dataset)

def encontrar_mejor_modelo(ruta_ejecucion="/content/runs/segment/"):
    """
    Objetivo:
        Permite encontrar el mejor modelo entrenado basándose en la métrica mAP50-95.

    Parámetros:
        runs_path (str): ruta donde se almacenan las carpetas de entrenamiento generadas por YOLO.

    Returns:
        str or None: ruta del mejor modelo encontrado o None si no se encuentra un modelo válido.
    """
    carpetas_entrenamiento = sorted(
        [archivo for archivo in os.listdir(ruta_ejecucion) if archivo.startswith("train") and os.path.isdir(os.path.join(ruta_ejecucion, archivo))],
        key=lambda x: int(x.replace("train", "")) if x.replace("train", "").isdigit() else 0
    )
    mejor_mAP = 0
    mejor_modelo = None
    for carpeta in carpetas_entrenamiento:
        carpeta_entrenamiento = os.path.join(ruta_ejecucion, carpeta)
        csv_resultados = os.path.join(carpeta_entrenamiento, "results.csv")
        ruta_mejor_modelo = os.path.join(carpeta_entrenamiento, "weights", "best.pt")
        if os.path.exists(csv_resultados) and os.path.exists(ruta_mejor_modelo):
            resultados = pd.read_csv(csv_resultados)
            if "metrics/mAP50-95(M)" in resultados.columns:
                max_mAP = resultados["metrics/mAP50-95(M)"].max()
                if max_mAP > mejor_mAP:
                    mejor_mAP = max_mAP
                    mejor_modelo = ruta_mejor_modelo
    return mejor_modelo

def predecir_con_mejor_modelo(ruta_mejor_modelo, directorio_datasets):
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
    diccionario_predicciones = {}
    for numero_dataset in range(1, 6):
        ruta_imagenes_test = os.path.join(directorio_datasets, f'Dataset{numero_dataset}/test/images')
        resultados = model.predict(ruta_imagenes_test, save=True, save_txt=True)
        for resultado in resultados:
            nombre_archivo = os.path.basename(resultado.path)
            if hasattr(resultado, 'boxes') and resultado.boxes is not None:
                coordenadas = resultado.boxes.xyxy.tolist()
            elif hasattr(resultado, 'masks') and resultado.masks is not None:
                coordenadas = resultado.masks.xy.tolist()
            else:
                coordenadas = []
            diccionario_predicciones[nombre_archivo] = coordenadas


def comprimir_y_descargar_resultados(carpeta_salida):
    """
    Objetivo:
        Permite comprimir los resultados de las predicciones en un archivo ZIP y lo descarga automáticamente.

    Parámetros:
        carpeta_salida: nombre del directorio o carpeta en el que se almacenarán los resultados.

    Returns:
        None: no devuelve nada.
    """
    shutil.make_archive(f'/content/{carpeta_salida}', 'zip', '/content/runs/segment')
    files.download(f'/content/{carpeta_salida}.zip')

def postprocesar_etiquetas(directorio_etiquetas = "/content/runs/segment/predict/labels"):
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


def ejecutar_pipeline():
    """
    Objetivo:
        Permite ejeuctar el proceso de entrenamiento y realizar la predicciones con el mejor modelo entrenado. Además permite postprocesar las coordenadas predichas y almacenar los resultados de interés. Para el correcto uso de este código, es esencial incluirlo en Google Colab.

    Parámetros:
        None: no recibe nada.

    Returns:
        None: no devuelve nada.
    """
    print ("Vinculando con Google Drive...")
    # Vincular con Google Drive
    montar_drive()

    print ("Extrayendo dataset...")
    # Tras haber metido los Datasets (comprimidos) en la carpeta TFG de mi Drive.
    ruta_zip = '/content/drive/MyDrive/TFG/Datasets_Comprimidos.zip'
    # Ruta donde se almacenarán los datasets extraídos
    directorio_datasets = '/content/Datasets'
    # Extracción de datasets
    extraer_datasets(ruta_zip, directorio_datasets)
    print ("Datasets extraidos.")
    
    print ("Iniciando entrenamiento...")
    # Ejecución del entrenamiendo-validación del modelo.
    ejecutar_entrenamiento(directorio_datasets)
    # Obtencion de la ruta del mejor modelo entrenado (por defecto la ruta donde se almacenan siempre)
    ruta_mejor_modelo = encontrar_mejor_modelo()
    print (f"El mejor modelo se encuentra en: {ruta_mejor_modelo}")
    print ("Entrenamiento finalizado")

    print ("Iniciando predicciones...")
    # Predicciones
    predecir_con_mejor_modelo(ruta_mejor_modelo, directorio_datasets)
    print ("Predicciones finalizadas.")

    print ("Almacenando resultados")
    # Resultados finales
    comprimir_y_descargar_resultados("ResultadoEntrenamiento_VC5Folds_Dataset125")
    
    # Postprocesado de etiquetas
    guardar_predicciones_postprocesadas()
    
    # Almacenaje de resultados
    shutil.make_archive(f'/content/entrenamientos_y_predicciones', 'zip', '/content/runs')
    files.download(f'/content/entrenamientos_y_predicciones.zip')
    shutil.make_archive(f'/content/predicciones_postprocesadas_final', 'zip', '/content/predicciones_postprocesadas')
    files.download(f'/content/predicciones_postprocesadas_final.zip')
    print ("Resultados almacenados")
    print ("Proceso finalizado")