import tensorflow as tf

from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
import os

#Aumenta el conjunto de datos de entrenamiento aplicando transformación de rotación, desplazamiento horizontal y vertical
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

def build_vgg16_model(input_shape=(64, 64, 3), num_classes=5, dense_unit=256, dropout_rate=0.3, learning_rate=0.001):
    # Cargar VGG16 sin la parte superior
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Congelar las capas convolucionales para evitar que se actualicen en la fase inicial
    for layer in conv_base.layers:
        layer.trainable = False

    # Construcción del modelo
    model = models.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(dense_unit, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')  # Capa de salida con softmax
    ])

    # Compilar el modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

from sklearn.utils import shuffle

def vgg16(X_train, y_train, X_test, y_test, class_names, batch_size=32, epochs=10):

    # Mezclar los datos antes de dividirlos en entrenamiento/validación
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Data augmentation con validación
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Divide en 80% entrenamiento y 20% validación
    )

    # Generadores de datos separados para entrenamiento y validación
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, subset="training", shuffle=True)
    val_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, subset="validation", shuffle=True)

    # Construcción del modelo
    num_classes = len(np.unique(y_train))
    model = build_vgg16_model(input_shape=(64, 64, 3), num_classes=num_classes)

    # Configurar early stopping
    earlystop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entrenamiento del modelo
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[earlystop_callback]
    )

    # Evaluación del rendimiento
    #evaluar_rendimiento(model, X_test, y_test, "VGG16")

    return model