import cv2
import os
from ultralytics import YOLO

model_path = "runs/detect/treino_epi_v1/weights/best.onnx"

if not os.path.exists(model_path):
    print(f"O arquivo {model_path} não foi encontrado!")
    exit()

print("Carregando modelos...")

model_people = YOLO('yolo11n.pt')

model_epi = YOLO(model_path)

VERDE = (0, 255, 0)
VERMELHO = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)

print("Sistema iniciado. Pressione 'q' para encerrar.")

while True:
    success, frame = cap.read()

    if not success:
        print("Erro ao abrir webcam")
        break

    results_people = model_people.predict(frame, classes=[0], conf=0.5, verbose=False)
    
    results_epi = model_epi.predict(frame, conf=0.5, verbose=False)

    boxes_people = results_people[0].boxes.xyxy.cpu().numpy()
    boxes_epi = results_epi[0].boxes.xyxy.cpu().numpy()

    for box_p in boxes_people:
        xp1, yp1, xp2, yp2 = box_p 
        
        esta_protegido = False

        for box_e in boxes_epi:
            xe1, ye1, xe2, ye2 = box_e 
            
            centro_epi_x = (xe1 + xe2) / 2
            centro_epi_y = (ye1 + ye2) / 2
            
            if (xp1 < centro_epi_x < xp2) and (yp1 - 50 < centro_epi_y < yp2):
                esta_protegido = True
                
                cv2.rectangle(frame, (int(xe1), int(ye1)), (int(xe2), int(ye2)), (0, 255, 255), 1)
                break 

        x1, y1, x2, y2 = int(xp1), int(yp1), int(xp2), int(yp2)

        if esta_protegido:
            cv2.rectangle(frame, (x1, y1), (x2, y2), VERDE, 2)
            cv2.rectangle(frame, (x1, y1-30), (x1+150, y1), VERDE, -1)
            cv2.putText(frame, "COM EPI", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), VERMELHO, 2)
            cv2.rectangle(frame, (x1, y1-30), (x1+150, y1), VERMELHO, -1) 
            cv2.putText(frame, "SEM EPI!", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Sistema de Monitoramento EPI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Programa encerrado.")