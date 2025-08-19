from google.colab import drive
drive.mount('/content/drive')

import os, time
import numpy as np
import matplotlib.pyplot as plt

filas, columnas = 1000, 1000
ruta_vector_drive = "/content/drive/MyDrive/vision/vector_1000x1000_uint8.txt"

import numpy as np, time
import matplotlib.pyplot as plt

inicio_lectura = time.perf_counter()
vector = np.loadtxt(ruta_vector_drive, dtype=np.uint8)
fin_lectura = time.perf_counter()

inicio_estadisticas = time.perf_counter()
minimo = int(vector.min())
maximo = int(vector.max())
media = float(vector.mean())
desviacion = float(vector.std())
fin_estadisticas = time.perf_counter()

print(f"Leídos: {vector.size} valores")
print(f"Mínimo: {minimo}  Máximo: {maximo}  Media: {media:.4f}  Desviación: {desviacion:.4f}")
print(f"Tiempo lectura: {fin_lectura - inicio_lectura:.3f}s")
print(f"Tiempo estadísticas NumPy: {fin_estadisticas - inicio_estadisticas:.3f}s")
print(f"Ruta usada: {ruta_vector_drive}")