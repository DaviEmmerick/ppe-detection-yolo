import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")

model_path = "runs/detect/treino_epi_v1/weights/best.pt"
model_name = "PPE-YOLOv11"

with mlflow.start_run(run_name="Registro-Limpo-Sem-Barra"):
    mlflow.log_artifact(model_path, artifact_path="weights")
    
    run_id = mlflow.active_run().info.run_id
    source_uri = f"mlflow-artifacts:/0/{run_id}/artifacts/weights"
    
    client = mlflow.tracking.MlflowClient()
    mv = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id
    )
    
    print(f"Sucesso! Versao {mv.version} criada.")