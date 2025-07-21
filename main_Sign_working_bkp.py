import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
import joblib

# Load model and label encoder
model = load_model("asl_model.h5")
encoder = joblib.load("label_encoder.pkl")

# Initialize TTS engine
engine = pyttsx3.init()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_letter = None

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add padding
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Extract hand ROI
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (28, 28))
            hand_img = hand_img.reshape(1, 28, 28, 1) / 255.0

            # Predict
            pred = model.predict(hand_img)
            class_index = np.argmax(pred)
            letter = encoder.classes_[class_index]

            # Show prediction
            cv2.putText(frame, f"Letter: {chr(letter + 65)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak only if changed
            if letter != prev_letter:
                engine.say(f"{chr(letter + 65)}")
                engine.runAndWait()
                prev_letter = letter

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
