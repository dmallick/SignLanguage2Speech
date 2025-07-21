import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from tensorflow.keras.models import load_model
import joblib

# Load model and label encoder
model = load_model("asl_model.h5")
encoder = joblib.load("label_encoder.pkl")

# TTS engine
engine = pyttsx3.init()

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

buffer = []               # Stores predicted letters
last_prediction = None    # Last letter seen
stable_counter = 0        # For debounce
prediction_threshold = 15 # Frames before confirming a letter

last_detected_time = time.time()
pause_threshold = 2       # seconds of hand absence = trigger TTS

def speak_word(word):
    print("Predicted Word:", word)
    engine.say(word)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    current_time = time.time()

    if result.multi_hand_landmarks:
        last_detected_time = current_time  # reset idle timer

        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (28, 28))
            hand_img = hand_img.reshape(1, 28, 28, 1) / 255.0

            pred = model.predict(hand_img)
            class_index = np.argmax(pred)
            letter = encoder.classes_[class_index]
            letter = chr(letter + 65)

            # Debounce logic
            if letter == last_prediction:
                stable_counter += 1
            else:
                stable_counter = 0
                last_prediction = letter

            # Add only if stable for enough frames
            if stable_counter == prediction_threshold:
                if not buffer or buffer[-1] != letter:
                    buffer.append(letter)
                    print("Buffer:", "".join(buffer))
                stable_counter = 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If hand missing for long enough, treat as pause → speak word
    elif buffer and (current_time - last_detected_time) > pause_threshold:
        word = "".join(buffer)
        speak_word(word)
        buffer = []

    # Display the buffer
    cv2.putText(frame, f"Word: {''.join(buffer)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Sign Language to Word", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
