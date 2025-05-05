from ultralytics import YOLO

# Carrega o modelo pré-treinado (pode usar yolov8n.pt, yolov8s.pt, etc.)
model = YOLO('yolo11n.pt')

# Treina o modelo com seu dataset
model.train(
    data='controle.yaml',     # caminho para o arquivo YAML
    epochs=60,            # número de épocas
)