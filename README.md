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

* **Linguagem:** Python
* **Modelo:** Ultralytics YOLOv11
* **Processamento de Imagem:** OpenCV
* **Hardware:** Treinamento realizado em GPU (Ideal)
* **Ferramenta de Anotação:** Roboflow

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

## 📊 Métricas e Resultados

Em breve