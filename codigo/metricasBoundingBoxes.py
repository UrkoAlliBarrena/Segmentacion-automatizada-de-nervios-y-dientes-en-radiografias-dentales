import os
import numpy as np

def obtener_cajas_delimitadoras(ruta):
    '''
    Objetivo: 
        Permite leer las coordenadas que delimitan los bordes de un objeto, de un archivo de texto(.txt), y las convierte
            en una lista de tuplas que representan cajas delimitadoras.
    
    Parámetros:
        ruta (str): ruta del archivo que tiene las coordenadas de las cajas delimitadoras.
    
    Returns:
        cajas_delimitadoras (list): lista de tuplas con las cajas delimitadoras.
    '''
    cajas_delimitadoras = []
    with open(ruta, 'r') as archivo:
        for linea in archivo:
            partes_linea = linea.strip().split()
            clase = int(partes_linea[0])
            coordenadas = list(map(float, partes_linea[1:]))
            coordenadas_x = coordenadas[0::2]
            coordenadas_y = coordenadas[1::2]
            coordenada_x_minima, coordenada_y_minima = min(coordenadas_x), min(coordenadas_y)
            coordenada_x_maxima, coordenada_y_maxima = max(coordenadas_x), max(coordenadas_y)
            cajas_delimitadoras.append((clase, 
                                        coordenada_x_minima, coordenada_y_minima, 
                                        coordenada_x_maxima, coordenada_y_maxima))
    return cajas_delimitadoras


def interseccion_sobre_union(caja1, caja2):
    '''
    Objetivo: 
        Permite calcular la Intersección sobre la Unión (IoU) entre dos cajas delimitadoras (estas cajas presentan
            una estructura definida: coordenada_x_esqIzq, coordenada_y_esqIzq, coordenada_x_esqDer, coordenada_y_esqDer.
            Siendo esqIzq, esquina izquierda, etc).
    
    Parámetros:
        caja1 (tuple): contiene las coordenadas de la primera caja.
        caja2 (tuple): contiene las coordenadas de la segunda caja.
    
    Returns:
        float: valor de la Intersección sobre la Unión entre 0 y 1, siempre que el área total sea mayor que 0.
    '''
    coordenada_x_esqIzq_1, coordenada_y_esqIzq_1, coordenada_x_esqDer_1, coordenada_y_esqDer_1 = caja1
    coordenada_x_esqIzq_2, coordenada_y_esqIzq_2, coordenada_x_esqDer_2, coordenada_y_esqDer_2 = caja2
    coordenada_izq_mayor = max(coordenada_x_esqIzq_1, coordenada_x_esqIzq_2)
    coordenada_sup_mayor = max(coordenada_y_esqIzq_1, coordenada_y_esqIzq_2)
    coordenada_der_mayor = min(coordenada_x_esqDer_1, coordenada_x_esqDer_2)
    coordenada_inf_mayor = min(coordenada_y_esqDer_1, coordenada_y_esqDer_2)
    ancho_interseccion = max(0, coordenada_der_mayor - coordenada_izq_mayor)
    alto_interseccion = max(0, coordenada_inf_mayor - coordenada_sup_mayor)
    area_interseccion = ancho_interseccion * alto_interseccion
    area_caja1 = (coordenada_x_esqDer_1 - coordenada_x_esqIzq_1) * (coordenada_y_esqDer_1 - coordenada_y_esqIzq_1)
    area_caja2 = (coordenada_x_esqDer_2 - coordenada_x_esqIzq_2) * (coordenada_y_esqDer_2 - coordenada_y_esqIzq_2)
    area_total = area_caja1 + area_caja2 - area_interseccion
    return area_interseccion/area_total if area_total > 0 else 0


def calcular_metricas(cajas_originales, cajas_predichas, umbral=0.5):
    '''
    Objetivo: 
        Permite calcular varias métricas como: precisión, recall, F1 y media de la IoU, para observar si las
            predicciones realizadas son adecuadas o no.
    
    Parámetros:
        cajas_originales (list): cajas delimitadoras originales (tienen el formato: (clase, x_min, y_min, x_max, y_max)).
        cajas_predichas (list): cajas delimitadoras predichas con el mismo formato.
        umbral (float): valor mínimo o umbral de IoU para considerar que una predicción es verdadera. Por defecto este 
            parámetro obtiene el valor de 0.5.
    
    Returns:
        (precision, recall, valor_F1, media_IoU) (tuple): tupla con los valores de cada métrica correspondiente.
    '''
    verdaderos_positivos = 0
    emparejados = set()
    valores_IoU = []
    for prediccion in cajas_predichas:
        mejor_IoU = 0
        indice_mejor_caja_original = -1
        for indice, caja in enumerate(cajas_originales):
            if indice in emparejados:
                continue
            actual_IoU = interseccion_sobre_union(prediccion[1:], caja[1:])
            if actual_IoU > mejor_IoU:
                mejor_IoU = actual_IoU
                indice_mejor_caja_original = indice
        if mejor_IoU >= umbral and indice_mejor_caja_original != -1:
            verdaderos_positivos += 1
            emparejados.add(indice_mejor_caja_original)
            valores_IoU.append(mejor_IoU)
    falsos_negativos = len(cajas_originales) - verdaderos_positivos
    falsos_positivos = len(cajas_predichas) - verdaderos_positivos
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
    valor_F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    media_IoU = np.mean(valores_IoU) if valores_IoU else 0
    return precision, recall, valor_F1, media_IoU


def obtener_metricas_por_clase(cajas_originales, cajas_predicciones, clases=[0, 1], umbral=0.5):
    '''
    Objetivo: 
        Permite calcular las métricas (precisión, recall, F1, IoU) para cada clase predicha.
    
    Parámetros:
        cajas_originales (list): cajas delimitadoras originales (tienen el formato: (clase, x_min, y_min, x_max, y_max)).
        cajas_predichas (list): cajas delimitadoras predichas con el mismo formato.
        clases (list): clases que se van a evaluar (Como sabemos que sólo hay dos, establecemos: [0,1]).
        umbral (float): valor mínimo o umbral de IoU para considerar que una predicción es verdadera. Por defecto este 
            parámetro obtiene el valor de 0.5.
            
    Returns:
        metricas (dict): se trata de un diccionario con las métricas calculadas individualmente para cada clase.
    '''
    metricas = {}
    for clase in clases:
        clase_original = [caja for caja in cajas_originales if caja[0] == clase]
        clase_prediccion = [caja for caja in cajas_predicciones if caja[0] == clase]
        precision, recall, valor_F1, media_IoU = calcular_metricas(clase_original, clase_prediccion, umbral)
        metricas[clase] = {"Precision": precision, "Recall": recall, "F1": valor_F1, "IoU": media_IoU}
    return metricas

def obtener_metricas(carpeta_coordenadas_originales, carpeta_coordenadas_predicciones, mostrar = True):
    '''
    Objetivo: 
        Permite calcular la media de cada métrica (precisión, recall, F1 e IoU) para ver si la calidad de las predicciones
            realizadas es la correcta.

    Parámetros:
        carpeta_coordenadas_originales (str): ruta de la carpeta con los archivos que contienen las coordenadas de
            las cajas delimitadoras originales (proporcionadas por el odontólogo).
        carpeta_coordenadas_predicciones (str): ruta de la carpeta con los archivos que contienen las coordenadas de las 
            cajas delimitadoras predichas.
        mostrar (bool): indica si se deben mostrar en pantalla los resultados. Por defecto toma el valor de True.

    Returns:
        None: no devuelve nada.
    '''
    archivos_comunes = set(os.listdir(carpeta_coordenadas_originales)).intersection(set(os.listdir(carpeta_coordenadas_predicciones)))
    resultados_por_archivo = {}
    for archivo in archivos_comunes:
        ruta_original = os.path.join(carpeta_coordenadas_originales, archivo)
        ruta_prediccion = os.path.join(carpeta_coordenadas_predicciones, archivo)
        caja_original = obtener_cajas_delimitadoras(ruta_original)
        caja_prediccion = obtener_cajas_delimitadoras(ruta_prediccion)
        metricas = obtener_metricas_por_clase(caja_original, caja_prediccion)
        resultados_por_archivo[archivo] = metricas
    media_metricas = {clase: {"Precision": 0, "Recall": 0, "F1": 0, "IoU": 0} for clase in [0,1]}
    numero_archivos = len(resultados_por_archivo)
    for metrica in resultados_por_archivo.values():
        for clase in [0,1]:
            media_metricas[clase]["Precision"] += metrica[clase]["Precision"]
            media_metricas[clase]["Recall"] += metrica[clase]["Recall"]
            media_metricas[clase]["F1"] += metrica[clase]["F1"]
            media_metricas[clase]["IoU"] += metrica[clase]["IoU"]
    for clase in [0,1]:
        media_metricas[clase]["Precision"] /= numero_archivos
        media_metricas[clase]["Recall"] /= numero_archivos
        media_metricas[clase]["F1"] /= numero_archivos
        media_metricas[clase]["IoU"] /= numero_archivos
    if mostrar == True:
        for clase, metrica in media_metricas.items():
            print(f"  Clase {'ROOT' if clase == 0 else 'TOOTH'}:")
            print(f"    Precision: {metrica['Precision']:.5f}")
            print(f"    Recall:    {metrica['Recall']:.5f}")
            print(f"    F1:        {metrica['F1']:.5f}")
            print(f"    IoU:       {metrica['IoU']:.5f}")

if __name__ == "__main__":
    print ("Obteniendo métricas...")
    obtener_metricas("Dataset\labels", "ENTRENAMIENTOS\SEG_VC_5FOLDS_PRUEBA\predict\labels")
    print ("Métricas obtenidas.")