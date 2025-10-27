import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1 Cấu hình cơ bản
# ==========================
MODEL_DIR = "model"        # Thư mục lưu mô hình
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_flowers.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100                 # Tăng lên (10–20) nếu muốn huấn luyện kỹ hơn

# 2 Tải Oxford Flowers 102 từ TensorFlow Datasets
# ==========================
print("Đang tải dataset Oxford 102 Flowers từ TensorFlow Datasets...")
import tensorflow_datasets as tfds

dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)

train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)

num_classes = info.features["label"].num_classes
class_names = info.features["label"].names
print(f"Đã tải dataset Oxford 102 Flowers với {num_classes} lớp hoa.")

# 3 Xây dựng mô hình MobileNetV2
# ==========================
print("🔹 Xây dựng mô hình MobileNetV2...")

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Đóng băng trọng số pre-trained

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4 Huấn luyện mô hình
# ==========================
print("🚀 Bắt đầu huấn luyện...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 5 Lưu mô hình
# ==========================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model.save(MODEL_PATH)
print(f"Mô hình đã được lưu tại: {MODEL_PATH}")

# 6 Đánh giá nhanh trên tập test
# ==========================
print("Đánh giá nhanh trên tập test:")
loss, acc = model.evaluate(test_ds)
print(f"🔹 Test accuracy: {acc:.4f}")