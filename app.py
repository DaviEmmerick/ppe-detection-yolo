from fastapi import File, UploadFile, FastAPI
from fastapi.responses import RedirectResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

@app.get('/')
def get_status():
    print("Acessaram a rota raiz. Redirecionando...")
    return RedirectResponse(url="/docs")


model_path = "runs/detect/treino_epi_v1/weights/best.onnx"
model_epi = YOLO(model_path)
model_people = YOLO("yolo11n.pt")

@app.post('/api/check_epi')
async def get_check(file: UploadFile = File(...)):
    print(f"1. Recebi uma chamada na API. Arquivo: {file.filename}")

    conteudo = await file.read()
    nparr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pessoas = model_people(img, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
    capacetes = model_epi(img, verbose=False)[0].boxes.xyxy.cpu().numpy()

    total_pessoas = len(pessoas)
    pessoas_protegidas = 0

    for xp1, yp1, xp2, yp2 in pessoas:
        for xc1, yc1, xc2, yc2 in capacetes:
            centro_capacete_x = (xc1 + xc2) / 2
            centro_capacete_y = (yc1 + yc2) / 2
            
            if (xp1 < centro_capacete_x < xp2) and (yp1 - 50 < centro_capacete_y < yp2):
                pessoas_protegidas += 1
                break

    status = "PERIGO"
    if total_pessoas > 0 and pessoas_protegidas == total_pessoas:
        status = "SEGURO"
    elif total_pessoas == 0:
        status = "VAZIO"

    return {
        "resultado": status,
        "pessoas_detectadas": total_pessoas,
        "pessoas_com_epi": pessoas_protegidas
    }