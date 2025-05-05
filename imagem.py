import os
from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO('runs/detect/train8/weights/last.pt')

# Caminho da pasta com imagens
img_dir = 'C:/Users/guich/Área de Trabalho/Controles/datasets/imagens'

# Itera sobre os arquivos da pasta
for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(img_dir, filename)

        # Carrega a imagem
        image = cv2.imread(img_path)
        if image is None:
            print(f"[AVISO] Não foi possível carregar a imagem: {filename}")
            continue

        # Faz a inferência
        results = model(img_path, conf=0.1)

        # Mostra a imagem com as detecções
        results[0].show()

        # Exibe os resultados no terminal
        print(f"[INFO] Resultados para {filename}:")
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"  {len(result.boxes)} item(ns) detectado(s):")
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"    - Classe: {model.names[cls_id]} | Confiança: {conf:.2f}")
            else:
                print("  Nenhum item detectado.")

