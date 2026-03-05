from flask import Flask, render_template, request
import os
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load SavedModel folder (NOT .keras file)
model = tf.keras.layers.TFSMLayer(
    "fall_detection_saved_model",
    call_endpoint="serving_default"
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            cap = cv2.VideoCapture(filepath)
            frames = []

            while len(frames) < 16:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) < 16:
                return "Video too short!"

            frames = np.array(frames)
            frames = np.expand_dims(frames, axis=0)

            # ✅ Predict using SavedModel
            prediction = model(frames)

            # SavedModel returns dictionary sometimes
            if isinstance(prediction, dict):
                prediction = list(prediction.values())[0]

            prediction = prediction.numpy()

            if prediction[0][0] > 0.5:
                return render_template("index.html", result="🚨 Fall Detected")
            else:
                return render_template("index.html", result="✅ No Fall Detected")

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)