from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def cnn1():
    model = models.Sequential()
    model.add(keras.Input(shape=(64, 64, 3)))  # Nueva dimensión de entrada

    # Primera capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Segunda capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling para reducir dimensionalidad
    model.add(layers.GlobalAveragePooling2D())

    # Capa densa
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout para evitar el sobreajuste

    # Capa de salida con 5 clases
    model.add(layers.Dense(5, activation='softmax'))

    # Compilación del modelo
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

def cnn2():
    model = models.Sequential()
    model.add(keras.Input(shape=(64, 64, 3)))  # Nueva dimensión de entrada

    # Bloque 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloque 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Global Average Pooling en lugar de Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Capa densa
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.2))

    # Capa de salida con Softmax
    model.add(layers.Dense(5, activation='softmax'))

    # Compilación del modelo
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model