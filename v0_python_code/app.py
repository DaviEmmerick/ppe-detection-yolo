from fastapi import File, UploadFile, FastAPI
from fastapi.responses import RedirectResponse
from ultralytics import YOLO
import mlflow
import cv2
import os
import numpy as np
from datetime import datetime

app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")


SAVE_DIR = "data/to_label"
os.makedirs(SAVE_DIR, exist_ok=True)
LOW_CONF_THRESHOLD = 0.55  

model_people = YOLO("yolo11n.pt")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/treino_epi_v1/weights/best.pt")

try:
    model_uri = "models:/PPE-YOLOv11@production"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    model_file = None
    if os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            for f in files:
                if f.endswith(".pt"):
                    model_file = os.path.join(root, f)
                    break
    else:
        model_file = local_path

    if not model_file or not os.path.exists(model_file):
        raise FileNotFoundError("Arquivo .pt nao encontrado no download")

    model_epi = YOLO(model_file)
    print(f"API Conectada ao MLflow: {model_file}")

except Exception as e:
    print(f"Falha MLflow: {e}")
    model_epi = YOLO(LOCAL_MODEL_PATH)

@app.get('/')
def get_status():
    return RedirectResponse(url="/docs")

@app.post('/api/check_epi')
async def get_check(file: UploadFile = File(...)):
    conteudo = await file.read()
    nparr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inferência
    res_pessoas = model_people(img, classes=[0], verbose=False)[0]
    pessoas = res_pessoas.boxes.xyxy.cpu().numpy()
    
    res_epi = model_epi(img, verbose=False)[0]
    capacetes = res_epi.boxes.xyxy.cpu().numpy()
    confidencias = res_epi.boxes.conf.cpu().numpy() 

    total_pessoas = len(pessoas)
    pessoas_protegidas = 0
    save_trigger = False

    for xp1, yp1, xp2, yp2 in pessoas:
        for xc1, yc1, xc2, yc2 in capacetes:
            cx, cy = (xc1 + xc2) / 2, (yc1 + yc2) / 2
            if (xp1 < cx < xp2) and (yp1 - 50 < cy < yp2):
                pessoas_protegidas += 1
                break

    if len(confidencias) > 0:
        if np.any(confidencias < LOW_CONF_THRESHOLD):
            save_trigger = True
    elif total_pessoas > 0:
        save_trigger = True

    if save_trigger:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f"{SAVE_DIR}/review_{timestamp}.jpg", img)

    status = "PERIGO"
    if total_pessoas > 0 and pessoas_protegidas == total_pessoas:
        status = "SEGURO"
    elif total_pessoas == 0:
        status = "VAZIO"

    return {
        "resultado": status,
        "pessoas_detectadas": total_pessoas,
        "pessoas_com_epi": pessoas_protegidas,
        "coletado_para_treino": save_trigger
    }