import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont
import os
from tensorflow.keras.applications.resnet import preprocess_input

# Font fallback paths
font_paths = [
    "NotoSansDevanagari-Regular.ttf",
    "fonts/NotoSansDevanagari-Regular.ttf",
    "C:/Windows/Fonts/mangal.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def find_working_font():
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, 40)
            except Exception:
                continue
    return ImageFont.load_default()

def render_text(frame, text, position, font):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def predict_sign(model_path, label_map_path):
    # Load model
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        print("❌ Model or label map not found.")
        return

    model = tf.keras.models.load_model(model_path)
    with open(label_map_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    index_to_label = {v: k for k, v in class_indices.items()}

    font = find_working_font()
    prediction_buffer = deque(maxlen=10)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to access webcam.")
        return

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]

                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)

                    pad = 20
                    x_min = max(x_min - pad, 0)
                    y_min = max(y_min - pad, 0)
                    x_max = min(x_max + pad, w)
                    y_max = min(y_max + pad, h)

                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size > 0:
                        try:
                            resized = cv2.resize(hand_roi, (128, 128))
                            input_img = preprocess_input(resized.astype(np.float32))
                            input_tensor = np.expand_dims(input_img, axis=0)

                            preds = model.predict(input_tensor, verbose=0)
                            pred_class = np.argmax(preds)
                            confidence = np.max(preds)

                            prediction_buffer.append(pred_class)
                            common_class = Counter(prediction_buffer).most_common(1)[0][0]

                            if confidence > 0.5:
                                label = index_to_label[common_class]
                                print(f"✅ Predicted: {label} (confidence: {confidence:.2f})")
                                frame = render_text(frame, f"{label} ({confidence:.2f})", (10, 50), font)
                            else:
                                cv2.putText(frame, "Low Confidence", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        except Exception as e:
                            print("Prediction Error:", e)
                            cv2.putText(frame, "Prediction Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            cv2.imshow("Nepali Sign Predictor (ResNet)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_sign("models/sign_language_model.h5", "models/class_indices.json")
