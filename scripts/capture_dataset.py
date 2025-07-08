import cv2
import os
import mediapipe as mp

def imwrite_unicode(filename, img):
    # Encode image as JPEG in memory buffer
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        print("❌ Encoding image failed")
        return False
    try:
        with open(filename, 'wb') as f:
            encoded_img.tofile(f)
        return True
    except Exception as e:
        print(f"❌ Exception saving file {filename}: {e}")
        return False

def capture_dataset(output_dir, label, num_images=200):
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    print(f"📸 Ready to capture {num_images} images for label: '{label}'")
    print("▶️ Press 'c' to start capturing, 'q' to quit")

    count = 0
    capturing = False

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = max(int(min(x_coords) * w) - 20, 0)
                x_max = min(int(max(x_coords) * w) + 20, w)
                y_min = max(int(min(y_coords) * h) - 20, 0)
                y_max = min(int(max(y_coords) * h) + 20, h)

                hand_roi = frame[y_min:y_max, x_min:x_max]

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if capturing and hand_roi.size > 0:
                    try:
                        hand_resized = cv2.resize(hand_roi, (128, 128))
                        file_path = os.path.join(label_dir, f"{label}_{count}.jpg")
                        success = imwrite_unicode(file_path, hand_resized)
                        if success:
                            print(f"[{count+1}/{num_images}] ✅ Saved: {file_path}")
                            count += 1
                        else:
                            print(f"❌ Failed to save image {file_path}")
                    except Exception as e:
                        print(f"⚠️ Exception while saving image: {e}")
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Label: {label}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Sign Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capturing = True
            print("🟢 Capturing started...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 Done capturing {count} images for label '{label}'")

if __name__ == "__main__":
    output_dir = "data/captured_signs"
    label = input("🔤 Enter the label for the sign (e.g., क): ")
    capture_dataset(output_dir, label)
