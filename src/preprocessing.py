import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'valid'),
        image_size=img_size,
        batch_size=batch_size
    )
    test_ds = image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=img_size,
        batch_size=batch_size
    )

    return train_ds, val_ds, test_ds
