import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from feature_engineering import augment_data

# 1️. Tải dataset Oxford 102 Flowers
# -------------------------
print("Tải dataset Oxford 102 Flowers...")
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features['label'].num_classes
print(f"Tổng số lớp hoa: {num_classes}")

# 2️. Tiền xử lý dữ liệu
# -------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Gọi hàm augment_data từ feature_engineering.py
ds_train = augment_data(ds_train)

ds_train = ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------
# 3️. Xây dựng mô hình MobileNetV2
# -------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Transfer learning

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# 4️. Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# -------------------------
os.makedirs("../102_flowers_classification_mobilenetv2/model", exist_ok=True)
checkpoint_path = "../102_flowers_classification_mobilenetv2/model/mobilenetv2_flowers.h5"

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# -------------------------
# 5️. Huấn luyện mô hình
# -------------------------
history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_val,
    callbacks=callbacks
)

# -------------------------
# 6️. Đánh giá và lưu mô hình
# -------------------------
test_loss, test_acc = model.evaluate(ds_test)
print(f"Accuracy trên tập test: {test_acc:.4f}")

model.save(checkpoint_path)
print(f"Mô hình đã được lưu tại: {checkpoint_path}")

# -------------------------
# 7️. Vẽ biểu đồ Accuracy & Loss
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training & Validation Accuracy")
plt.legend()
plt.savefig("../102_flowers_classification_mobilenetv2/web/static/training_accuracy.png")

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("../102_flowers_classification_mobilenetv2/web/static/training_loss.png")

# -------------------------
# 8️. Confusion Matrix (10 loại hoa đầu)
# -------------------------
y_true = []
y_pred = []

for images, labels in ds_test.take(50):  # Lấy ngẫu nhiên 50 batch test
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(cm[:10, :10])  # chỉ hiển thị 10 loại đầu
plt.figure(figsize=(8, 8))
cm_display.plot(cmap='viridis', colorbar=False)
plt.title("Confusion Matrix - 10 lớp hoa đầu")
plt.savefig("../102_flowers_classification_mobilenetv2/web/static/confusion_matrix.png")
plt.close()

print("Huấn luyện hoàn tất, biểu đồ đã lưu trong thư mục web/static/")
