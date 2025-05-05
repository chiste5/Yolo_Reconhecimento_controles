from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO('runs/detect/train8/weights/last.pt')

# Caminho da imagem
img_path = 'C:/Users/guich/Área de Trabalho/Controles/datasets/imagens/PlayStation-5.jpg'

# Faz a inferência
results = model(img_path, conf=0.1)

# Carrega a imagem original
image = cv2.imread(img_path)

# Desenha as detecções
# Inference
results = model(img_path, conf=0.1)

# Mostra a imagem com os resultados
results[0].show()

# Também imprime os resultados
for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"[INFO] {len(result.boxes)} item(ns) detectado(s)!")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - Classe: {model.names[cls_id]} | Confiança: {conf:.2f}")
    else:
        print("[INFO] Nenhum item detectado.")

