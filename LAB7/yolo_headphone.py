from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO('best.pt')

# Abrir la cámara (0 = webcam por defecto)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la predicción sobre el frame
    results = model(frame, conf=0.5)

    # Dibujar las detecciones en el frame
    annotated_frame = results[0].plot()

    # Mostrar en pantalla
    cv2.imshow("YOLO Cam", annotated_frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
