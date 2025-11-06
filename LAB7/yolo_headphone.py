from ultralytics import YOLO
model = YOLO('/home/carlos/IA/LAB6/LAB6.2/TRAIN_YOLO/best.pt')
results = model.predict(source=0,  # webcam o cámara conectada
                        imgsz=640,
                        conf=0.5)
results.show()