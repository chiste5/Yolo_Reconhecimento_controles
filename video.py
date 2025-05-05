from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO('runs/detect/train8/weights/last.pt')

# Abre a webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar o frame.")
        break

    # Faz a inferência no frame
    results = model(frame, conf=0.05)

    # Mostra o frame com as detecções
    annotated_frame = results[0].plot()  # Desenha as caixas
    cv2.imshow('YOLO Detection', annotated_frame)

    # Imprime os resultados
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"[INFO] {len(result.boxes)} item(ns) detectado(s)!")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  - Classe: {model.names[cls_id]} | Confiança: {conf:.2f}")
        else:
            print("[INFO] Nenhum item detectado.")

    # Sai do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
