import tensorflow as tf

def augment_data(dataset):
    """
    Thực hiện Data Augmentation chuyên nghiệp cho hình ảnh hoa.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def augment(image, label):
        image = data_augmentation(image)
        return image, label

    return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
