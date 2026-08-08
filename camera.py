import cv2
import csv
import time
import os
import mediapipe as mp

#  Config do path e MediaPipe Tasks
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
CSV_FILE = os.path.join(SCRIPT_DIR, "dataset_libras.csv")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexões das articulações da mão
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# 2. Inicialização do arquivo CSV para o Dataset
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# Configuração do detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# sinal para gravar
CURRENT_LABEL = "C"

webcam = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print(f"Gravando para o sinal: '{CURRENT_LABEL}'")

    while webcam.isOpened():
        success, frame = webcam.read()

        if not success:
            print("Falha ao capturar imagem.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(time.time() * 1000)
        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Lista para armazenar as coordenadas normalizadas para o CSV
        amostra_normalizada = []

        if resultado.hand_landmarks:
            altura, largura, _ = frame.shape

            # Processa a primeira mão detectada para a amostra
            hand_landmarks = resultado.hand_landmarks[0]

            # Ponto 0 (punho)
            wrist = hand_landmarks[0]

            # Extração com normalização
            for lm in hand_landmarks:
                amostra_normalizada.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            # Desenho na tela
            for hand_idx, hand in enumerate(resultado.hand_landmarks):
                # Desenha os pontos
                for landmark in hand:
                    x = int(landmark.x * largura)
                    y = int(landmark.y * altura)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Desenha as linhas dos dedos
                for inicio, fim in CONNECTIONS:
                    p1 = (int(hand[inicio].x * largura), int(hand[inicio].y * altura))
                    p2 = (int(hand[fim].x * largura), int(hand[fim].y * altura))
                    cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.2)

                # Vetor do punho para a palma (ponto 0-ponto 9)
                w_pt = (int(hand[0].x * largura), int(hand[0].y * altura))
                p_pt = (int(hand[9].x * largura), int(hand[9].y * altura))
                cv2.arrowedLine(frame, w_pt, p_pt, (255, 255, 0), 4, tipLength=0.3)

        # Informações na tela
        cv2.putText(frame, f"Sinal: '{CURRENT_LABEL}'", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Hand Landmarker - Libras Collector", frame)

        tecla = cv2.waitKey(1) & 0xFF

        # aperte s para gravar no CSV
        if tecla == ord('s') or tecla == ord('S'):
            if amostra_normalizada:
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([CURRENT_LABEL] + amostra_normalizada)
                print(f"[OK] Amostra salva para o sinal '{CURRENT_LABEL}'")
            else:
                print("[!] Nenhuma mão detectada para salvar.")

        # Pressionar q ou esx para sair
        elif tecla == ord("q") or tecla == 27:
            break

webcam.release()
cv2.destroyAllWindows()