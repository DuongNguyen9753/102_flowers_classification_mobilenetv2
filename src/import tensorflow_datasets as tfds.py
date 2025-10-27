import tensorflow_datasets as tfds

# Tải metadata dataset Oxford 102 Flowers
_, info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
class_names = info.features["label"].names

# Lưu danh sách tên hoa vào file .txt
output_path = "oxford_102_flower_labels.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for i, name in enumerate(class_names):
        f.write(f"{i}: {name}\n")

print("✅ File danh sách tên hoa đã lưu tại:", output_path)