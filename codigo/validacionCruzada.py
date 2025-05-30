import os
import shutil
import numpy as np
import zipfile
from sklearn.model_selection import KFold

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

def obtener_rutas_imagenes_y_coordenadas(ruta_imagenes, ruta_coordenadas):
    """
    Objetivo: 
        Permite obtener las rutas de las imágenes y sus coordenadas correspondientes.
    
    Parámetros:
        ruta_imagenes (str): ruta de la carpeta que contiene las imágenes.
        ruta_coordenadas (str): ruta de la carpeta que contiene las coordenadas referentes al
                                canal radicular y muela de cada imagen.
    
    Returns:
        x (np.array): array con las rutas de las imágenes.
        y (np.array): array con las rutas de las coordenadas referentes a las imágenes.
    """
    x, y = [], []
    for nombre_imagen in os.listdir(ruta_imagenes):
        ruta_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        ruta_coordenada = os.path.join(ruta_coordenadas, nombre_imagen.replace(".png", ".txt"))
        if os.path.exists(ruta_coordenada):
            x.append(ruta_imagen)
            y.append(ruta_coordenada)
    return np.array(x), np.array(y)

def validacion_cruzada(x, y, numero_folds=5):
    """
    Objetivo: 
        Permite realizar la división de datos en entrenamiento, validación y prueba usando la
        técnica de validación cruzada.
    
    Parámetros:
        x (np.array): array con las rutas de las imágenes.
        y (np.array): array con las rutas de las coordenadas referentes a las imágenes.
        numero_folds (int): número de particiones que se van a realizar en la validación cruzada.
    
    Returns:
        datos_Folds (list): lista con los datos divididos en cada fold.
    """
    objeto_KFold = KFold(n_splits=numero_folds, shuffle=True)
    datos_Folds = []
    for indices_entrenamiento, indices_validacion in objeto_KFold.split(x):
        X_entrenamiento, X_intermedio = x[indices_entrenamiento], x[indices_validacion]
        Y_entrenamiento, Y_intermedio = y[indices_entrenamiento], y[indices_validacion]
        X_validacion, X_test = np.array_split(X_intermedio, 2)
        Y_validacion, Y_test = np.array_split(Y_intermedio, 2)
        datos_Folds.append((X_entrenamiento, Y_entrenamiento,
                            X_validacion, Y_validacion,
                            X_test, Y_test))
    return datos_Folds

def copiar_archivos(rutas_imagenes, rutas_coordenadas, carpeta_destino):
    """
    Objetivo: 
        Permite copiar los archivos de imágenes y coordenadas en su destino correspondiente.
    
    Parámetros:
        rutas_imagenes (list): lista con las rutas de cada una de las imágenes.
        rutas_coordenadas (list): lista con las rutas de las coordenadas respectivas al canal
                                  radicular y muela de cada imagen.
        carpeta_destino (str): carpeta destino donde se van a copiar los archivos.
    
    Returns: 
        None: no devuelve nada.
    """
    for ruta_imagen, ruta_coordenadas in zip(rutas_imagenes, rutas_coordenadas):
        shutil.copy(ruta_imagen, os.path.join(carpeta_destino, "images", os.path.basename(ruta_imagen)))
        shutil.copy(ruta_coordenadas, os.path.join(carpeta_destino, "labels", os.path.basename(ruta_coordenadas)))

def guardar_yaml(carpeta_salida, numero_fold):
    """
    Objetivo:
        Permite almacenar un archivo YAML con la estructura especificada para el dataset.

    Parámetros:
        carpeta_salida (str): carpeta donde se guardará el archivo YAML.
        numero_fold (int): número que representa el fold que se está realizando.

    Returns: 
        None: no devuelve nada.
    """
    datos_yaml = {
        'train': f'/content/Datasets/Dataset{numero_fold}/train/images',
        'val': f'/content/Datasets/Dataset{numero_fold}/val/images',
        'test': f'/content/Datasets/Dataset{numero_fold}/test/images',
        'nc': 2,
        'names': ['ROOT', 'TOOTH']
    }
    with open(os.path.join(carpeta_salida, f'dataset{numero_fold}.yaml'), 'w') as archivo_yaml:
        for clave, valor in datos_yaml.items():
            archivo_yaml.write(f"{clave}: {valor}\n")
        
def generar_dataset(datos_Folds, numero_folds=5):
    """
    Objetivo: 
        Permite crear las carpetas donde se van a almacenar los archivos, y permite copiar 
        los archivos obtenidos en cada fold.
    
    Parámetros:
        datos_Folds (list): lista con los datos obtenidos en cada fold.
        numero_folds (int): número de folds que se van a realizar.
    
    Returns: 
        None: no devuelve nada.
    """
    for fold, (X_entrenamiento, Y_entrenamiento, X_validacion, Y_validacion, X_test, Y_test) in enumerate(datos_Folds, start=1):
        carpeta_dataset = f"Dataset{fold}"
        for tipo_carpeta in ["train", "val", "test"]:
            crear_directorio(os.path.join(carpeta_dataset, tipo_carpeta, "images"))
            crear_directorio(os.path.join(carpeta_dataset, tipo_carpeta, "labels"))
        copiar_archivos(X_entrenamiento, Y_entrenamiento, os.path.join(carpeta_dataset, "train"))
        copiar_archivos(X_validacion, Y_validacion, os.path.join(carpeta_dataset, "val"))
        copiar_archivos(X_test, Y_test, os.path.join(carpeta_dataset, "test"))
        guardar_yaml(carpeta_dataset, fold)
        
def comprimir_datasets(zip_salida, numero_folds=5):
    """
    Objetivo: 
        Permite comprimir los datasets generados en un único archivo ZIP.
    
    Parámetros:
        zip_salida (str): nombre del archivo ZIP de salida que se va a generar.
        numero_folds (int): número de folds generador, y que se van a comprimir.
    
    Returns: 
        None: no devuelve nada.
    """
    carpetas_para_comprimir = [f"Dataset{i}" for i in range(1, numero_folds + 1)]
    with zipfile.ZipFile(zip_salida, "w", zipfile.ZIP_DEFLATED) as archivo_zip:
        for carpeta in carpetas_para_comprimir:
            for ruta_actual, subcarpetas_actuales, archivos in os.walk(carpeta):
                for archivo in archivos:
                    ruta_archivo = os.path.join(ruta_actual, archivo)
                    nombre_archivo = os.path.relpath(ruta_archivo, start=os.path.dirname(carpeta))
                    archivo_zip.write(ruta_archivo, nombre_archivo)

def eliminar_datasets(numero_folds=5):
    """
    Objetivo: 
        Permite eliminar los directorios o carpetas de los datasets generados en el proceso de 
        validación cruzada, con el objetivo de ahorrar en memoria.
    
    Parámetros:
        numero_folds (int): número de folds que se han generado.
    
    Returns: 
        None: No devuelve nada.
    """
    for fold in range(1, numero_folds+1):
        carpeta_dataset = f"Dataset{fold}"
        if os.path.exists(carpeta_dataset):
            try:
                shutil.rmtree(carpeta_dataset)
            # El siguiente código viene dado, debido a que hay algunas veces que al ejecutar esta
            # función aparece: PermissionError: [WinError 5] Acceso denegado: 'Dataset1\\test\\images'.
            # Por lo que si quiero eliminar los datasets generados tengo que cambiar los permisos
            # antes de intentar eliminar los datasets.
            except PermissionError:
                os.chmod(carpeta_dataset, 0o777)
                shutil.rmtree(carpeta_dataset)

if __name__ == "__main__":
    x, y = obtener_rutas_imagenes_y_coordenadas("Dataset/images", "Dataset/labels")
    print ("Iniciando validación cruzada...")
    datos_Folds = validacion_cruzada(x, y)
    print ("Obteniendo conjunto de datos...")
    generar_dataset(datos_Folds)
    comprimir_datasets("Datasets_Comprimidos.zip")
    eliminar_datasets()
    print ("Proceso finalizado, resultados en la carpeta 'Datasets_Comprimidos.zip'.")