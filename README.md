# 👁️ PPE Detection Project: YOLOv11

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-purple)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green)

## 📄 Sobre o Projeto

Este projeto consiste no treinamento e implantação de um modelo de Visão Computacional baseado na arquitetura **YOLO (You Only Look Once)** para a detecção automática de **PPE e equipamentos de segurança**.

O objetivo é automatizar a identificação de objetos em tempo real (ou em vídeos gravados), garantindo alta precisão e velocidade.

### 🎯 Objetivos

* Coletar e anotar um dataset customizado de imagens.
* Treinar o modelo YOLOv11 para identificar as classes.
* Validar a performance utilizando métricas como mAP (mean Average Precision).
* Criar script de inferência para uso em imagens e vídeos.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python, C++
* **Modelo:** Ultralytics YOLOv11
* **Processamento de Imagem:** OpenCV
* **Hardware:** Treinamento realizado em GPU (Ideal)
* **MLOps:** FastAPI, MLflow, Docker, AWS

## 📂 Estrutura do Projeto

```text
yolo-project/
├── data/                   # Arquivo data.yaml e estrutura de pastas (train/val/test)
├── weights/                # Pesos treinados (best.pt, last.pt)
├── inference/              # Scripts de teste e detecção
├── training_yolo.ipynb     # Notebook de treinamento
├── requirements.txt        # Dependências
└── README.md
```


## 🐳 Como rodar com Docker

Para garantir que o ambiente tenha todas as dependências de Visão Computacional, utilize o Docker:

1. **Build da imagem:**
   ```bash
   docker build -t api-epi-v1 .
   ```

2. **Run do container**
   ```bash
    docker run -p 8000:8000 api-epi-v1
   ```

## 📊 Métricas e Resultados (V0)

Abaixo estão os resultados do treinamento inicial realizado com YOLO11. 
Este modelo serve como baseline para a migração para C++.

![Resultados do Treinamento](results.png)

**Destaques Técnicos:**
* **mAP50:** Atingiu ~0.8, demonstrando alta confiabilidade na localização dos EPIs.
* **Estabilidade:** Curvas de Loss de validação seguem o treino, indicando ausência de overfitting.

## 🚀 Roadmap de Evolução (V1)

Atualmente o projeto está em sua fase de prototipagem (Python). Os próximos passos focam em performance e escalabilidade industrial:

- [ ] **Migração para C++:** Reescrever o pipeline de inferência para reduzir latência.
- [ ] **Otimização de Modelo:** Conversão para ONNX/TensorRT com quantização FP16/INT8.
- [ ] **Deploy Cloud (AWS):** Implementação de pipeline de CD para AWS ECR/ECS.
- [ ] **Monitoramento (MLOps):** Tracking de experimentos com MLflow.
