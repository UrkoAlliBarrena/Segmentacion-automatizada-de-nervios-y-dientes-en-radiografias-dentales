# https://imageio.readthedocs.io/en/stable/examples.html
import os
import imageio.v2 as iio

def obtenerDimensionesImagenes(ruta, salida):
    '''
    Funcionalidad:
      Permite calcular las dimensiones de cada una de las imágenes almacenadas en la ruta especificada ("ruta_imagenes").

    Parámetros:
      - ruta (string): ruta del directorio que contiene las imágenes.
      - salida (string): ruta del archivo en el que se van a encontrar las dimensiones para cada imagen.

    Returns:
      - None: no devuelve nada.
    '''
    numero_carpeta=0
    with open(salida, 'w') as archivo:
        for subcarpeta in os.listdir(ruta):
            ruta_subcarpeta = os.path.join(ruta, subcarpeta)
            if os.path.isdir(ruta_subcarpeta):
                for nombre_archivo in os.listdir(ruta_subcarpeta):
                    if nombre_archivo.lower().endswith(('.jpg', 'png')):
                        ruta_imagen = os.path.join(ruta_subcarpeta, nombre_archivo)
                        imagen = iio.imread(ruta_imagen)
                        alto, ancho = imagen.shape[:2]
                        archivo.write(f"{subcarpeta}: {ancho}px x {alto}px\n")
                archivo.write("\n")
            numero_carpeta+=1
        archivo.write(f"Número total de carpetas: {numero_carpeta}")

def ejecutar_pipeline():
    '''
    Funcionalidad:
      Permite ejecutar el pipeline encargado de obtener las dimensiones de cada una de las imágenes de una carpeta.

    Parámetros:
      - None: no recibe nada.

    Returns:
      - None: no devuelve nada.
    '''
    print ("Obteniendo dimensiones...")
    ruta = "ConjuntoDatosOdontologo\MÁSTER BENJAMÍN MARTÍN BIEDMA"
    salida = "dimensiones.txt"
    obtenerDimensionesImagenes(ruta, salida)
    print (f"Dimensiones obtenidas en: {salida}")

if __name__ == "__main__":
    ejecutar_pipeline()