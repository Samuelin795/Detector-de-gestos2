import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Inicializamos MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Archivo para guardar el modelo entrenado
MODEL_FILE = "gestures_model.pkl"

# Función para guardar el modelo
def save_model(data, labels):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

# Función para cargar el modelo
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model["data"], model["labels"]
    return [], []

# Modo de entrenamiento
def train_gestures():
    data, labels = load_model()
    print("Modo de entrenamiento activado.")
    print("Presiona 's' para capturar un gesto, 'q' para salir.")
    
    label = input("Introduce el nombre del gesto que deseas entrenar: ")
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesamiento de la imagen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Obtenemos los puntos clave normalizados
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    cv2.putText(frame, "Press 's' to save gesture", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        print(f"Gesto '{label}' capturado.")
                        data.append(landmarks)
                        labels.append(label)

            cv2.imshow("Entrenamiento", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    save_model(data, labels)

# Modo de detección
def detect_gestures():
    data, labels = load_model()
    if not data:
        print("No hay gestos entrenados. Entrena primero un gesto.")
        return

    print("Modo de detección activado.")
    print("Presiona 'q' para salir.")

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesamiento de la imagen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Obtenemos los puntos clave normalizados
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    # Comparación con los gestos entrenados
                    distances = [np.linalg.norm(np.array(landmarks) - np.array(d)) for d in data]
                    min_index = np.argmin(distances)
                    detected_label = labels[min_index]

                    cv2.putText(frame, f"Gesto: {detected_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Detección", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Menú principal
def main():
    while True:
        print("\n--- Menú ---")
        print("1. Entrenar gestos")
        print("2. Detectar gestos")
        print("3. Salir")

        choice = input("Selecciona una opción: ")
        if choice == '1':
            train_gestures()
        elif choice == '2':
            detect_gestures()
        elif choice == '3':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
