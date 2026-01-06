import os
import mlflow
from ultralytics import YOLO
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
MODEL_NAME = "PPE-YOLOv11"
MAP_THRESHOLD = 0.55
DATA_YAML = "dataset/data.yaml" 

def run_monitoring_pipeline():
    client = MlflowClient()
    
    try:
        model_uri = f"models:/{MODEL_NAME}@production"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        
        model_file = local_path
        if os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for f in files:
                    if f.endswith(".pt"):
                        model_file = os.path.join(root, f)
                        break
        
        model = YOLO(model_file)
        print(f"Modelo carregado com sucesso: {model_file}")

    except Exception as e:
        print(f"Erro ao buscar modelo no MLflow: {e}")
        return

    results = model.val(data=DATA_YAML, verbose=False)
    current_map = results.results_dict['metrics/mAP50(B)']
    
    print(f"mAP50 Atual: {current_map:.4f}")

    if current_map < MAP_THRESHOLD:
        print(f"ALERTA: mAP abaixo do limite de {MAP_THRESHOLD}. Iniciando treinamento...")
        
        with mlflow.start_run(run_name=f"Retrain_mAP_{current_map:.2f}"):
            model.train(
                data=DATA_YAML, 
                epochs=30, 
                imgsz=640, 
                batch=16,
                name="retrain_output"
            )
            
            new_model_path = "runs/detect/retrain_output/weights/best.pt"
            mlflow.log_artifact(new_model_path, artifact_path="weights")
            
            run_id = mlflow.active_run().info.run_id
            model_uri_new = f"runs:/{run_id}/weights/best.pt"
            mv = client.create_model_version(name=MODEL_NAME, source=model_uri_new, run_id=run_id)
            
            print(f"\n[SUCESSO] Nova versão {mv.version} registrada.")
            print("Verifique os resultados no MLflow UI antes de promover para @production.")
    else:
        print("Performance estável. O modelo atual continua em produção.")

if __name__ == "__main__":
    run_monitoring_pipeline()