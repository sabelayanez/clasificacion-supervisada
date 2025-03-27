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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import validacion

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

def vgg16(X_train, y_train_encoded, X_test, y_test_encoded, class_names, batch_size=32, epochs=10):

    # Mezclar los datos antes de dividirlos en entrenamiento/validación
    X_train, y_train_encoded = shuffle(X_train, y_train_encoded, random_state=42)

    # Data augmentation con validación
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Divide en 80% entrenamiento y 20% validación
    )

    # Generadores de datos separados para entrenamiento y validación
    train_generator = train_datagen.flow(X_train, y_train_encoded, batch_size=batch_size, subset="training", shuffle=True)
    val_generator = train_datagen.flow(X_train, y_train_encoded, batch_size=batch_size, subset="validation", shuffle=True)

    # Construcción del modelo
    num_classes = len(np.unique(y_train_encoded))
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

    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=1)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    classification_rep = classification_report(y_test_encoded, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test_encoded, y_pred)
    auc_roc = roc_auc_score(y_test_encoded, y_pred_prob, multi_class='ovr', average='macro')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", cm)
    print(f"AUC-ROC Score: {auc_roc:.4f}")


    #validacion(X_test, y_test_encoded, y_pred, class_names)
    # Evaluación del rendimiento
    #evaluar_rendimiento(model, X_test, y_test, "VGG16")

    return model