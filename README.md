# 🤟 Nepali Sign Language Recognition (Devanagari Letters)

This project uses computer vision and deep learning to recognize **Nepali Sign Language** hand signs from a webcam in real-time. It supports **36 Devanagari characters** (`क` to `ज्ञ`) and displays the recognized character in the video stream.

---

## 📌 Features

- 🔍 Real-time hand detection using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- 🤖 Sign classification using a fine-tuned **ResNet50** CNN
- 🧠 Prediction smoothing using a buffer for stability
- 📸 Displays Devanagari predictions **inside the hand ROI box**
- 🔡 Supports rendering **Unicode Devanagari fonts** correctly

---

## 📂 Project Structure

.
├── models/
│ ├── sign_language_model.h5 # Trained ResNet50 model
│ └── class_indices.json # Label map for Devanagari characters
├── data/
│ └── captured_signs/ # Dataset directory (optional)
├── Predict Sign Updated.py # Main prediction script (live webcam)
├── train_model.py # CNN training script with augmentation
├── .gitignore
└── README.md

yaml
Copy
Edit

---

## 🧪 Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- MediaPipe
- Pillow
- NumPy
- Matplotlib (for training visualization)

Install dependencies:

```bash
pip install -r requirements.txt
Tip: Use mangal.ttf or NotoSansDevanagari-Regular.ttf in the fonts/ folder for proper Devanagari rendering.

🚀 Running the Project
🔍 1. Predict Hand Signs
bash
Copy
Edit
python "Predict Sign Updated.py"
Press q to quit the webcam view.

Make sure your webcam is accessible and the model path is correct.

🧠 2. Train on Your Own Dataset (Optional)
To train on your custom hand signs:

bash
Copy
Edit
python train_model.py
Edit data_dir, model_save_path, and label_map_path in the script as needed.

📝 Dataset Format
Organize your dataset as:

kotlin
Copy
Edit
data/
└── captured_signs/
    ├── क/
    │   ├── क_0.jpg
    │   ├── क_1.jpg
    │   └── ...
    ├── ख/
    └── ...
Use your webcam to collect images into folders named with Devanagari letters.

✅ Model Highlights
📈 Uses ResNet50 pretrained on ImageNet with frozen base + custom classification head

🎛️ Applies advanced data augmentation (brightness, rotation, shift, etc.)

🧪 Includes early stopping + best checkpoint saving

🎨 Displays Devanagari prediction using PIL with fallback font logic

👀 Output Preview
ROI box drawn around detected hand

Predicted label shown in green text inside the box

🧠 Future Work
🕵️ Switch to landmark-based LSTM model for dynamic gestures

📱 Export model to TensorFlow Lite for mobile deployment

🌐 Create a web version using TensorFlow.js or MediaPipe Web

🧑‍💻 Author
Parichit Giri

📜 License
This project is open source and available under the MIT License.

yaml
Copy
Edit

---

Let me know if you'd like to add screenshots, badge support, or a link to a video demo!