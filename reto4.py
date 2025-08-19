import cv2
import numpy as np

# Captura de la cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Rangos de colores en HSV
rangos_colores = {
    "Rojo": [(0, 120, 70), (10, 255, 255)],
    "Verde": [(40, 40, 40), (70, 255, 255)],
    "Azul": [(90, 50, 50), (130, 255, 255)],
    "Rosado": [(140, 50, 50), (170, 255, 255)],
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in rangos_colores.items():
        lower = np.array(lower)
        upper = np.array(upper)

        # Máscara para detectar el color
        mask = cv2.inRange(hsv, lower, upper)

        # Encontrar contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 500:  # evitar ruido
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (255, 255, 255), 2)

    # Mostrar resultado
    cv2.imshow("Seguimiento de Colores", frame)

    # Salir con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
