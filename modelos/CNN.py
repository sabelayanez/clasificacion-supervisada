from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import evaluar_rendimiento

import numpy as np

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

def cnn_cross_validation(X_train, y_train_encoded, X_test, y_test_encoded, cnn_model='cnn1', CV=5, epochs=10, batch_size=32):
    # Definir el número de folds
    CV = 5
    kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)


    # Para almacenar las métricas de cada fold
    scores = {
        "precision_macro": [],
        "recall_macro": [],
        "precision_micro": [],
        "recall_micro": [],
        "f1_macro": [],
        "accuracy": [],
        "roc_auc": []
    }

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train_encoded)):
        print(f"\nFold {fold + 1}/{CV}")

        # Dividir los datos en train y validación
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train_encoded[train_index], y_train_encoded[val_index]

        # Crear un nuevo modelo en cada fold
        if cnn_model == 'cnn1':
            model = cnn1()
        else:
            model = cnn2()

        # Entrenar el modelo
        model.fit(X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=1)

        # Predecir en el conjunto de validación
        y_pred_probs = model.predict(X_val_fold)  # Probabilidades
        y_pred = np.argmax(y_pred_probs, axis=1)  # Clases predichas

        # Calcular métricas
        scores["accuracy"].append(accuracy_score(y_val_fold, y_pred))
        scores["precision_macro"].append(precision_score(y_val_fold, y_pred, average="macro"))
        scores["recall_macro"].append(recall_score(y_val_fold, y_pred, average="macro"))
        scores["precision_micro"].append(precision_score(y_val_fold, y_pred, average="micro"))
        scores["recall_micro"].append(recall_score(y_val_fold, y_pred, average="micro"))
        scores["f1_macro"].append(f1_score(y_val_fold, y_pred, average="macro"))

        # Calcular ROC AUC (solo si es clasificación multiclase con one-hot encoding)
        try:
            roc_auc = roc_auc_score(y_val_fold, y_pred_probs, multi_class="ovr")
            scores["roc_auc"].append(roc_auc)
        except ValueError:
            scores["roc_auc"].append(None)

    # Mostrar los resultados de cada fold
    for metric, values in scores.items():
        print(f"\n{metric}: {values}")

    evaluar_rendimiento(
        y_val_fold,
        y_pred_probs,
        y_pred,
        cnn_model
    )