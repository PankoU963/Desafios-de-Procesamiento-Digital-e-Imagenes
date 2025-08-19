import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------
# Función para añadir ruido sal y pimienta
# ------------------------
def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    h, w = image.shape[:2]
    for i in range(h):
        for j in range(w):
            rdn = random.random()
            if rdn < prob:        # Sal (blanco)
                noisy[i][j] = 255
            elif rdn > 1 - prob:  # Pimienta (negro)
                noisy[i][j] = 0
    return noisy

# ------------------------
# Convolución manual
# ------------------------
def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    # Padding
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255)

# ------------------------
# Filtros
# ------------------------
def mean_filter(image, ksize=3):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return convolution(image, kernel)

def gaussian_filter(image, ksize=3, sigma=1):
    ax = np.linspace(-(ksize-1)/2., (ksize-1)/2., ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return convolution(image, kernel)

def median_filter(image, ksize=3):
    h, w = image.shape
    pad = ksize // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+ksize, j:j+ksize].flatten()
            output[i, j] = np.median(region)
    
    return output

# ------------------------
# MAIN
# ------------------------
# Leer imagen en escala de grises
image = cv2.imread("imagen_limpia.jpg", cv2.IMREAD_GRAYSCALE)

# Añadir ruido
noisy = add_salt_pepper_noise(image, prob=0.02)

# Aplicar filtros manuales
mean_img = mean_filter(noisy, 3)
gaussian_img = gaussian_filter(noisy, 5, sigma=1)
median_img = median_filter(noisy, 3)

# Filtros de OpenCV para comparar
opencv_median = cv2.medianBlur(noisy, 3)

# ------------------------
# Mostrar resultados
# ------------------------
titles = ["Original", "Con Ruido", "Media (manual)", "Gaussiano (manual)", "Mediana (manual)", "Mediana OpenCV"]
images = [image, noisy, mean_img, gaussian_img, median_img, opencv_median]

plt.figure(figsize=(12,6))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
