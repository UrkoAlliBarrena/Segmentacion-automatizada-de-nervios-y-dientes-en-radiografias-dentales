import os
from PIL import Image, ImageDraw

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

def leer_etiqueta(ruta_coordenadas):
    """
    Objetivo: 
        Permite leer las coordenadas de un archivo para organizarlas según su categoría.
        
    Parámetros:
        ruta_coordenadas (str): ruta del archivo que contiene las etiquetas y coordenadas de los contornos.
        
    Returns:
        coordenadas_etiquetadas (dict): diccionario con las coordenadas organizadas en listas según su etiqueta.
    """
    coordenadas_etiquetadas = {"0": [], "1": []}
    with open(ruta_coordenadas, 'r') as coordenadas:
        for linea in coordenadas:
            partes = linea.strip().split()
            etiqueta = partes[0]
            coordenadas = [(float(partes[i]), float(partes[i+1])) for i in range(1, len(partes), 2)]
            if etiqueta in coordenadas_etiquetadas:
                coordenadas_etiquetadas[etiqueta].append(coordenadas)
    return coordenadas_etiquetadas

def dibujar_contornos(imagen, contornos, ruta_salida):
    """
    Objetivo: 
        Permite dibujar los contornos sobre una imagen a partir de las coordenadas (normalizadas), y guarda la imagen 
        en la ruta especificada.
        
    Parámetros:
        imagen (PIL.Image): imagen sobre la que se va a dibujar el contorno de la muela y el canal radicular.
        contornos (dict): diccionario con las coordenadas de los contornos según sus etiquetas.
        ruta_salida (str): ruta donde se almacenarán las imágenes contorneadas.
        
    Returns: 
        None: no devuelve nada.
    """
    dibujar = ImageDraw.Draw(imagen)
    colores = {"0": "red", "1": "blue"}
    grosor = 3
    for etiqueta, lista_contornos in contornos.items():
        color = colores.get(etiqueta, "green")
        for contorno in lista_contornos:
            if len(contorno) >= 3:
                contorno_normal = [(x * imagen.width, y * imagen.height) for x, y in contorno]
                contorno_normal.append(contorno_normal[0])
                dibujar.line(contorno_normal, fill=color, width=grosor)
    imagen.save(ruta_salida)

def procesar_entrenamiento(carpeta_entrenamiento, carpeta_salida):
    """
    Objetivo: 
        Permite procesar las imágenes de entrenamiento, para leer sus etiquetas y dibujar sus contornos.
        
    Parámetros:
        carpeta_entrenamiento (str): ruta de la carpeta que contiene las imágenes y coordenadas de entrenamiento.
        carpeta_salida (str): ruta donde se almacenarán las imágenes con los contornos dibujados.
    Returns: None
    """
    carpeta_imagenes = os.path.join(carpeta_entrenamiento, "images")
    carpeta_coordenadas = os.path.join(carpeta_entrenamiento, "labels")
    crear_directorio(carpeta_salida)
    for archivos in os.listdir(carpeta_imagenes):
        if archivos.lower().endswith(('.jpg', '.png')):
            nombre_imagen = os.path.splitext(archivos)[0]
            ruta_imagen = os.path.join(carpeta_imagenes, archivos)
            ruta_coordenadas = os.path.join(carpeta_coordenadas, f"{nombre_imagen}.txt")
            if os.path.exists(ruta_coordenadas):
                imagen = Image.open(ruta_imagen).convert("RGB")
                contornos = leer_etiqueta(ruta_coordenadas)
                ruta_salida_imagen = os.path.join(carpeta_salida, archivos)
                dibujar_contornos(imagen, contornos, ruta_salida_imagen)

if __name__ == "__main__":
    ruta_base = r"Dataset"
    ruta_salida = r"coordenadasConfirmadas"
    print ("Iniciando la confirmación de las coordenadas normalizadas...")
    procesar_entrenamiento(ruta_base, ruta_salida)
    print ("Proceso de confirmación de coordenadas normalizadas terminado")