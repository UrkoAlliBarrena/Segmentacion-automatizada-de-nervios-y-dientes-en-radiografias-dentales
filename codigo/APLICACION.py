#!pip install customtkinter
#!python.exe -m pip install --upgrade pip
#!pip install --upgrade numpy scipy ultralytics opencv-python
#!pip install "numpy<2.0"

import os
import threading

import customtkinter
from tkinter import filedialog, Tk, messagebox
from PIL import Image
import numpy as np

import REALIZAR_PREDICCION


class Aplicacion(customtkinter.CTk):
    def __init__(self):
        """
        Funcionalidad:
          Permite inicializar la aplicación y configurar la interfaz. Además, define variables de estado.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        super().__init__()
        self.configurar_interfaz()
        self.imagen_seleccionada = None
        self.prediccion = None
        self.protocol("WM_DELETE_WINDOW", self.cerrar_ventana)

    def cerrar_ventana(self):
        """
        Funcionalidad:
          Intenta cerrar la ventana principal de la aplicación, y en caso de que no se pueda, muestra un mensaje de error.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        try:
            self.destroy()
        except:
            ventana_emergente = Tk()
            ventana_emergente.withdraw()
            messagebox.showinfo("Error", "⚠️ La ventana no se puede cerrar")

    def configurar_interfaz(self):
        """
        Funcionalidad:
          Permite configurar la interfaz de la aplicación, en consecuencia, al configurar la ventana principal, 
          permite crear botones y paneles para mostrar las imágenes y los resultados.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        self.title("🦷 RootScan AI - Detección Inteligente de Canales Radiculares")
        self.geometry("1600x750")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.configure(padx=20, pady=20)

        self.boton_cargar = customtkinter.CTkButton(self, text="CARGAR IMAGEN", command=self.cargar_archivo,font=("Arial", 16, "bold"), corner_radius=5, width=250)
        self.boton_cargar.grid(row=0, column=0, pady=(10, 10), sticky="n")

        self.boton_ejecutar = customtkinter.CTkButton(self, text="CALCULAR LONGITUD", command=self.ejecutar_prediccion,font=("Arial", 16, "bold"), corner_radius=5, width=250)
        self.boton_ejecutar.grid(row=1, column=0, pady=(10, 30), sticky="n")

        self.boton_guardar = customtkinter.CTkButton(self, text="GUARDAR RESULTADO", command=self.guardar_resultados,font=("Arial", 16, "bold"), corner_radius=5, width=250)
        self.boton_guardar.grid(row=3, column=0, pady=(10, 30), sticky="n")

        self.grid_columnconfigure(1, minsize=100)
        self.panel_original = self.crear_panel("RADIOGRAFÍA ORIGINAL", 2)
        self.grid_columnconfigure(3, minsize=50)
        self.panel_mascara = self.crear_panel("CANAL RADICULAR", 4)
        self.grid_columnconfigure(5, minsize=50)
        self.panel_longitud = self.crear_panel("ESQUELETO", 6)
        self.grid_columnconfigure(7, minsize=50)

        self.pixeles_totales = customtkinter.CTkLabel(self, text="PÍXELES CANAL RADICULAR: SIN VALOR", font=("Arial", 18, "bold"), text_color="lightblue", anchor="w")
        self.pixeles_totales.grid(row=5, column=2, columnspan=4, pady=(30, 0), sticky="w")

        self.longitud = customtkinter.CTkLabel(self, text="LONGITUD CANAL RADICULAR: SIN VALOR", font=("Arial", 18, "bold"), text_color="lightblue", anchor="w")
        self.longitud.grid(row=6, column=2, columnspan=4, pady=(30, 0), sticky="w")

    def crear_panel(self, titulo, columna):
        """
        Funcionalidad:
          Permite crear un panel con título y el área para mostrar una imagen.

        Parámetros:
          - self: instancia de la clase Aplicacion.
          - titulo (str): nombre o encabezado del panel que hace referencia a la imagen que se muestra debajo.
          - columna (int): índice de columna donde se va a situar el panel creado.

        Returns:
          - panel_imagen (CTkLabel): panel en el que se mostrará la imagen con su encabezado.
        """
        nombre = customtkinter.CTkLabel(self, text=titulo, font=("Arial", 16, "bold"), text_color="#CCCCCC", anchor="center")
        nombre.grid(row=0, column=columna, pady=(0, 5))
        panel_imagen = customtkinter.CTkLabel(self, text="")
        panel_imagen.grid(row=1, column=columna, padx=10, pady=10)
        return panel_imagen

    def cargar_archivo(self):
        """
        Funcionalidad:
          Permite abrir una ventana para que el usuario pueda seleccionar una imagen, y la carga.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        ruta_imagen = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if ruta_imagen:
            self.imagen_seleccionada = ruta_imagen
            self.after(0, lambda: self.obtener_imagen(ruta_imagen, self.panel_original))

    def obtener_imagen(self, ruta_imagen, panel_imagen):
        """
        Funcionalidad:
          Permite cargar una imagen para su posterior visualización.

        Parámetros:
          - self: instancia de la clase Aplicacion.
          - ruta_imagen (str o array): ruta del archivo o array de píxeles.
          - panel_imagen (CTkLabel): panel o lugar donde se mostrará la imagen.

        Returns:
          - None: no devuelve nada.
        """
        imagen = Image.open(ruta_imagen) if isinstance(ruta_imagen, str) else Image.fromarray(ruta_imagen.astype(np.uint8))
        self.after(0, lambda: self._mostrar_imagen(imagen, panel_imagen))

    def _mostrar_imagen(self, imagen, panel_imagen):
        """
        Funcionalidad:
          Permite redimensionar el tamaño de la imagen si esta es demasiado grande y mostrar la imagen en el panel.

        Parámetros:
          - self: instancia de la clase Aplicacion.
          - imagen (PIL.Image): imagen que se va a mostrar.
          - panel_imagen (CTkLabel): panel o lugar donde se mostrará la imagen.

        Returns:
          - None: no devuelve nada.
        """
        dimensiones = (250, 400) if imagen.size[0] >= 720 else imagen.size
        imagen_tkinter = customtkinter.CTkImage(light_image=imagen, dark_image=imagen, size=dimensiones)
        panel_imagen.configure(image=imagen_tkinter, text="")
        panel_imagen.image = imagen_tkinter

    def ejecutar_prediccion(self):
        """
        Funcionalidad:
          Permite realizar una predicción sin bloquear la interfaz.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        self.prediccion = threading.Thread(target=self.realizar_prediccion, daemon=True)
        self.prediccion.start()

    def guardar_resultados(self):
        """
        Funcionalidad:
          Permite abrir una ventana para que el usuario pueda seleccionar una carpeta y guarda las imágenes generadas.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta para guardar los resultados")
        nombre_base_imagen = os.path.splitext(os.path.basename(self.imagen_seleccionada))[0]
        paneles = [self.panel_mascara, self.panel_longitud]
        imagenes = ["_segmentacion.png", "_longitud.png"]
        for panel, tipo_imagen in zip(paneles, imagenes):
            if hasattr(panel, "image") and panel.image:
                imagen_original = panel.image._light_image
                ruta_guardado = os.path.join(carpeta, nombre_base_imagen + tipo_imagen)
                imagen_original.save(ruta_guardado)
        #Nuevo
        '''
        texto_long = self.longitud.cget("text")
        valor = re.sub(r"[^\d\.,]", "", texto_long).replace(",", ".")
        longitud_predicha = f"Longitud: {valor} mm"
        ruta_excel = os.path.join(carpeta, "Results.xlsx")
        REALIZAR_PREDICCION.registrar_resultado_en_excel(self.imagen_seleccionada, longitud_predicha, ruta_excel)
        '''

        
    def realizar_prediccion(self):
        """
        Funcionalidad:
          Permite ejecutar el flujo completo de predicción, postprocesado y muestra resultados.

        Parámetros:
          - self: instancia de la clase Aplicacion.

        Returns:
          - None: no devuelve nada.
        """
        ruta = self.imagen_seleccionada
        nombre_imagen = os.path.splitext(os.path.basename(ruta))[0] + ".png"

        REALIZAR_PREDICCION.predecir_con_mejor_modelo("best.pt", ruta)
        REALIZAR_PREDICCION.guardar_predicciones_postprocesadas()

        directorio = os.path.dirname(ruta)
        REALIZAR_PREDICCION.postprocesado("runs/segment/predict/labels", directorio, "FINAL")

        mascara = REALIZAR_PREDICCION.procesar_imagen(os.path.join("FINAL", nombre_imagen), "MASCARAS_FINAL")
        self.after(0, lambda: self.obtener_imagen(mascara, self.panel_mascara))

        # NUEVO
        esqueleto = REALIZAR_PREDICCION.obtener_imagenes_skeleton("MASCARAS_FINAL", "SKELETON")
        esqueleto_grises = esqueleto[..., 0] if esqueleto.ndim == 3 else esqueleto
        mascara_esqueleto = (esqueleto_grises == 255) if esqueleto_grises.max() > 1 else (esqueleto_grises == 1)
        altura, anchura = esqueleto_grises.shape
        forma_esqueleto = np.zeros((altura, anchura, 3), dtype=np.uint8)
        forma_esqueleto[mascara_esqueleto] = [255, 255, 255]
        pixeles_blancos = int(mascara_esqueleto.sum())
        self.after(0, lambda: self.pixeles_totales.configure(text=f"PÍXELES CANAL RADICULAR: {pixeles_blancos}"))
        self.after(0, lambda img=forma_esqueleto: self.obtener_imagen(img, self.panel_longitud))

        longitud = REALIZAR_PREDICCION.obtener_longitud_esqueleto(esqueleto)
        self.after(0, lambda: self.longitud.configure(text=f"LONGITUD CANAL RADICULAR: {longitud} milímetros"))

        REALIZAR_PREDICCION.eliminar_carpetas()

if __name__ == "__main__":
    app = Aplicacion()
    app.mainloop()
    
### Bibliografía necesaria
# https://docs.python.org/3/library/tkinter.html
# https://customtkinter.tomschimansky.com/
# https://docs.python.org/3/library/threading.html
# https://docs.python.org/3/library/tkinter.messagebox.html
# https://docs.python.org/3/library/dialog.html