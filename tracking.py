import mlflow
from ultralytics import YOLO
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PPE-DETECTION-V0")

model_relative_path = "runs/detect/treino_epi_v1/weights/best.pt"

with mlflow.start_run(run_name="Tracking - V0"):
    model = YOLO(model_relative_path)

    mlflow.autolog()
    mlflow.log_param("model_type", "YOLOv11n")
    mlflow.log_param("framework", "PyTorch")

    if os.path.exists(model_relative_path):
        mlflow.log_artifact(model_relative_path)
        print("Modelo registrado no MLflow com sucesso!")
    else:
        print(f"ERRO: Arquivo não encontrado em: {os.path.abspath(model_relative_path)}")