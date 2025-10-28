import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
app = Flask(__name__,
            template_folder="../web/templates",
            static_folder="../web/static")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = r"..\102_flowers_classification_mobilenetv2\model\mobilenetv2_flowers.h5"
STATIC_DIR = os.path.join(BASE_DIR, "../web/static/uploads")
CHART_DIR = os.path.join(BASE_DIR, "../web/static/charts")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model MobileNetV2 đã sẵn sàng.")

# ---------------- LOAD FLOWER LABELS ----------------
_, info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
class_names = info.features["label"].names

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    """Trang chủ: upload ảnh để dự đoán"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Nhận file upload và trả về kết quả dự đoán"""
    file = request.files['file']
    if not file:
        return render_template('index.html', prediction="Vui lòng chọn ảnh hoa để dự đoán.")

    # Lưu ảnh upload
    img_path = os.path.join(STATIC_DIR, file.filename)
    file.save(img_path)
    rel_path = f"uploads/{file.filename}"

    # Tiền xử lý
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Dự đoán
    preds = model.predict(img_array)
    pred_id = np.argmax(preds)
    confidence = np.max(preds)
    flower_name = class_names[pred_id]

    # Vẽ biểu đồ top 5 dự đoán
    top5_idx = np.argsort(preds[0])[-5:][::-1]
    top5_labels = [class_names[i] for i in top5_idx]
    top5_conf = preds[0][top5_idx]
    plt.figure(figsize=(6,4))
    plt.barh(top5_labels[::-1], top5_conf[::-1])
    plt.xlabel("Confidence")
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "top5.png")
    plt.savefig(chart_path)
    plt.close()

    rel_chart = "charts/top5.png"

    return render_template('index.html',
                           prediction=f"{flower_name} ({confidence*100:.2f}%)",
                           image_path=rel_path,
                           chart_path=rel_chart)

@app.route('/dashboard')
def dashboard():
    """Trang phân tích mô hình"""
    # Ở đây bạn có thể tạo biểu đồ độ chính xác, loss, confusion matrix v.v.
    # Ví dụ: biểu đồ giả lập
    plt.figure()
    epochs = list(range(1, 11))
    acc = [0.65, 0.70, 0.76, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90]
    plt.plot(epochs, acc, marker='o')
    plt.title("Accuracy theo epochs (giả lập)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    chart_path = os.path.join(CHART_DIR, "training_acc.png")
    plt.savefig(chart_path)
    plt.close()

    rel_chart = "charts/training_acc.png"
    return render_template('dashboard.html', chart_path=rel_chart)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
