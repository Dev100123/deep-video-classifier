import tensorflow as tf

class VideoModel:
    @staticmethod
    def build(input_shape=(10, 224, 224, 3)):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling3D(),
            tf.keras.layers.Conv3D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling3D(),
            tf.keras.layers.Conv3D(128, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling3D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
        ])

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"]
        )
        return model
