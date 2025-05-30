import os
import json
from PIL import Image

def crear_directorio(carpeta):
    '''
    Objetivo: 
        Permite crear un directorio o carpeta con el nombre especificado si esta no existe.
    
    Parámetros:
        carpeta (str): Nombre del directorio o carpeta que se va a crear.
    
    Returns:
        None: no devuelve nada
    '''
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

def leer_json(ruta_json):
    '''
    Objetivo: 
        Permite leer un archivo JSON y devolver el contenido que almacena.
    
    Parámetros:
        ruta_json (str): representa la ruta del archivo JSON.
    
    Returns:
        datos (dict): Contiene la información del archivo JSON leido en forma de diccionario.
    '''
    with open(ruta_json, 'r') as archivo_json:
        datos_json = json.load(archivo_json)
    return datos_json

def guardar_imagen(imagen, ruta_salida):
    '''
    Objetivo: 
        Permite almacenar una imagen en la ruta especificada.
    
    Parámetros:
        imagen (PIL.Image.Image): es la imagen que se va a guardar.
        ruta_salida (str): representa la ruta donde se va a guardar la imagen.
    
    Returns:
        None: no devuelve nada.
    '''
    if isinstance(imagen, str):
        imagen = Image.open(imagen)
    imagen.save(ruta_salida)
    
def guardar_coordenadas(coordenadas, ruta_salida):
    """
    Objetivo: 
        Permite almacenar las coordenadas normalizadas.
        
    Parámetros:
        coordenadas (str): coordenadas normalizadas.
        ruta_salida (str): ruta del archivo de salida.
        
    Returns: 
        None: no devuelve nada.
    """
    with open(ruta_salida, 'w') as archivo_coordenadas:
        archivo_coordenadas.write(coordenadas)

def normalizar_coordenadas(coordenadas_x, coordenadas_y, ancho, alto):
    '''
    Objetivo: 
        Permite normalizar las coordenadas en 'x' e 'y' en función de las dimensiones de cada imagen.
    
    Parámetros:
        coordenadas_x (list): lista con las coordenadas del eje 'x' del contorno.
        coordenadas_y (list):  lista con las coordenadas del eje 'y' del contorno.
        ancho (int): ancho de la imagen.
        alto (int): alto de la imagen.
    
    Returns:
        coordenadas_x_normalizados, coordenadas_y_normalizados(tuple): tupla con las coordenadas 'x' e 'y' normalizadas.
    '''
    coordenadas_x_normalizados = [x / ancho for x in coordenadas_x]
    coordenadas_y_normalizados = [y / alto for y in coordenadas_y]
    return coordenadas_x_normalizados, coordenadas_y_normalizados

def procesar_subcarpeta(ruta_subcarpeta, carpeta_casos):
    """
    Objetivo: 
        Procesar una subcarpeta con imágenes y JSON de coordenadas,
        acumulando en carpeta_casos las anotaciones normalizadas.

    Parámetros:
        ruta_subcarpeta (str): ruta de la subcarpeta con datos.
        carpeta_casos (list): lista donde se almacenan [ruta_imagen, anotaciones].

    Returns: 
        list: carpeta_casos actualizada con todas las anotaciones.
    """
    for nombre_archivo in os.listdir(ruta_subcarpeta):
        if not nombre_archivo.lower().endswith(".json"):
            continue
        ruta_json = os.path.join(ruta_subcarpeta, nombre_archivo)
        datos_json = leer_json(ruta_json)
        for valores_json in datos_json.values():
            regiones = valores_json.get("regions", {})
            regiones_iter = regiones.values() if isinstance(regiones, dict) else regiones
            nombre_imagen = valores_json.get("filename", "")
            ruta_imagen = os.path.join(ruta_subcarpeta, nombre_imagen)
            if not os.path.exists(ruta_imagen):
                partes = nombre_imagen.split('. ', 1)
                if len(partes) == 2:
                    sufijo = partes[1]
                    candidatos = [f for f in os.listdir(ruta_subcarpeta) if f.endswith(sufijo)]
                    if candidatos:
                        ruta_imagen = os.path.join(ruta_subcarpeta, candidatos[0])
            imagen = Image.open(ruta_imagen)
            ancho, alto = imagen.size
            etiqueta = "1" if "TOOTH" in ruta_json.upper() else "0"
            for idx, region in enumerate(regiones_iter):
                if not isinstance(region, dict):
                    continue
                forma = region.get("shape_attributes", {})
                xs = forma.get("all_points_x", [])
                ys = forma.get("all_points_y", [])
                if not xs or not ys:
                    continue
                xs_norm, ys_norm = normalizar_coordenadas(xs, ys, ancho, alto)
                pares = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs_norm, ys_norm))
                linea = f"{etiqueta} {pares}"
                for caso in carpeta_casos:
                    if caso[0] == ruta_imagen:
                        caso[1] += "\n" + linea
                        break
                else:
                    carpeta_casos.append([ruta_imagen, linea])

    return carpeta_casos
    
def obtener_dataset_completo(casos_totales, carpeta_salida):
    """
    Objetivo: 
        Permite generar el dataset con imágenes y etiquetas en carpetas separadas. Es decir, el 'Dataset' se encontrará 
        formado por dos subcarpetas: 'images' y 'labels'.
    
    Parámetros:
        casos_totales (list): lista de tuplas con las rutas de las imágenes y sus coordenadas correspondientes normalizadas.
        carpeta_salida (str): ruta de la carpeta donde se almacenará el dataset generado.
    
    Returns: 
        None: no devuelve nada.
    """
    carpetas = {
        "images": os.path.join(carpeta_salida, "images"),
        "labels": os.path.join(carpeta_salida, "labels")
    }
    for carpeta in carpetas.values():
        crear_directorio(carpeta)
    for i, (ruta_imagen, coordenadas) in enumerate(casos_totales, 1):
        nombre_imagen = f"CASO_{i}.png"
        ruta_destino_imagen = os.path.join(carpetas["images"], nombre_imagen)
        imagen = Image.open(ruta_imagen)
        guardar_imagen(imagen, ruta_destino_imagen)
        nombre_coordenadas = f"CASO_{i}.txt"
        ruta_destino_coordenadas = os.path.join(carpetas["labels"], nombre_coordenadas)
        guardar_coordenadas(coordenadas, ruta_destino_coordenadas)

def guardar_yaml(carpeta_salida):
    """
    Objetivo:
        Permite almacenar un archivo YAML con la estructura especificada para el dataset.

    Parámetros:
        carpeta_salida (str): carpeta donde se guardará el archivo YAML.

    Returns: 
        None: no devuelve nada.
    """
    datos_yaml = {
        # Ajusto las rutas para trabajar en Google Colab
        'train': '/content/Dataset/train/images',
        'val': '/content/Dataset/val/images',
        'test': '/content/Dataset/test/images',
        'nc': 2,
        'names': ['ROOT', 'TOOTH']
    }
    with open(os.path.join(carpeta_salida, 'dataset.yaml'), 'w') as archivo_yaml:
        for clave, valor in datos_yaml.items():
            archivo_yaml.write(f"{clave}: {valor}\n")
                    
def transformacion_coordenadas(carpeta, carpeta_salida, porc_entrenamiento=0.7):
    """
    Objetivo: 
        Permite transformar las imágenes y sus correspondientes coordenadas obtenidas de una carpeta, y seguidamente,
        dividirlas en conjuntos de entrenamiento y validación.
        
    Parámetros:
        carpeta (str): ruta de la carpeta con los datos originales.
        carpeta_salida (str): ruta o nombre de la carpeta de salida.
        porc_entrenamiento (float): proporción de datos para entrenamiento (por defecto 0.7).
        
    Returns: 
        None: no devuelve nada.
    """
    crear_directorio(carpeta_salida)
    casos = []
    for subcarpeta in os.listdir(carpeta):
        ruta_subcarpeta = os.path.join(carpeta, subcarpeta)
        if os.path.isdir(ruta_subcarpeta):
            casos_completos = procesar_subcarpeta(ruta_subcarpeta, casos)
    #dividir_datos(casos_completos, porc_entrenamiento, carpeta_salida)
    print ("Obteniendo dataset completo...")
    obtener_dataset_completo(casos_completos, carpeta_salida)
    print ("Dataset completo obtenido")
    print ("Obteniendo archivo '.yaml' ...")
    guardar_yaml(carpeta_salida)
    print ("Archivo '.yaml' obtenido")

if __name__ == "__main__":
    transformacion_coordenadas('ConjuntoDatosOdontologo\MÁSTER BENJAMÍN MARTÍN BIEDMA', 'Dataset')
    print ("Finalizado")