import cv2
import os
from ultralytics import YOLO

model_path = "runs/detect/treino_epi_v1/weights/best.pt"

if not os.path.exists(model_path):
  print(f"O arquivo {model_path} não foi encontrado!")

else:
  print("Modelo encontrado, iniciando webcam")

  model = YOLO(model_path)
  cap = cv2.VideoCapture(0)
  cap.set(3, 640)
  cap.set(4, 480)
  print("Pressione 'q' para encerrar ")

  while True:
    success, frame = cap.read()

    if not success:
      print("Erro ao abrir webcam")
      break

    results = model.predict(frame, conf=0.5, verbose=False)
    frame_anotado = results[0].plot()
    cv2.imshow("EPI Test", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()
  print("Programa encerrado.")