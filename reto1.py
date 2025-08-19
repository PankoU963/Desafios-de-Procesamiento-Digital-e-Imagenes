# reto1_local.py
import random, time, math
import matplotlib.pyplot as plt

filas, columnas = 1000, 1000
ruta_vector = "vector_1000x1000_uint8.txt"

def crear_matriz(filas, columnas, minimo=0, maximo=255):
    return [[random.randint(minimo, maximo) for _ in range(columnas)] for _ in range(filas)]

def calcular_estadisticas(matriz):
    minimo, maximo = 255, 0
    conteo, media, acumulador = 0, 0.0, 0.0  # Welford
    for fila in matriz:
        for valor in fila:
            if valor < minimo: minimo = valor
            if valor > maximo: maximo = valor
            conteo += 1
            delta = valor - media
            media += delta / conteo
            acumulador += delta * (valor - media)
    varianza = acumulador / conteo
    desviacion = math.sqrt(varianza)
    return minimo, maximo, media, desviacion

def aplanar_matriz(matriz):
    return [valor for fila in matriz for valor in fila]

def guardar_vector(vector, ruta):
    with open(ruta, "w") as f:
        for v in vector: f.write(f"{v}\n")

# tiempos y ejecución
inicio_generacion = time.perf_counter()
matriz = crear_matriz(filas, columnas)
fin_generacion = time.perf_counter()

inicio_estadisticas = time.perf_counter()
minimo, maximo, media, desviacion = calcular_estadisticas(matriz)
fin_estadisticas = time.perf_counter()

vector = aplanar_matriz(matriz)
guardar_vector(vector, ruta_vector)

print(f"Dimensión: {filas}x{columnas} = {filas*columnas} píxeles")
print(f"Mínimo: {minimo}  Máximo: {maximo}  Media: {media:.4f}  Desviación: {desviacion:.4f}")
print(f"Tiempo generación: {fin_generacion - inicio_generacion:.3f}s")
print(f"Tiempo estadísticas: {fin_estadisticas - inicio_estadisticas:.3f}s")

print(f"Archivo guardado: {ruta_vector}")

plt.imshow(matriz, cmap="gray", vmin=0, vmax=255)
plt.title("Matriz aleatoria 1000x1000")
plt.axis("off")
plt.show()