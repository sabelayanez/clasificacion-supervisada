from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from utils import validacion, extract_hog_features, evaluar_rendimiento
from constants import epochs

def ann_hog(X_train, y_train, X_test, y_test_encoded, num_classes, class_names, plot):
    # Convertir las etiquetas a one-hot encoding
    
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)
    
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test_encoded, num_classes)

    # Crear el modelo ANN
    ann_model = Sequential([
        Dense(512, activation='relu', input_dim=X_train_hog.shape[1]),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # num_classes = cantidad de clases
    ])

    # Compilar el modelo
    ann_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo ANN
    ann_model.fit(X_train_hog, y_train_onehot, epochs=epochs, batch_size=32, validation_data=(X_test_hog, y_test_onehot))

    # Obtener predicciones
    y_pred_prob = ann_model.predict(X_test_hog)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Obtener clases predichas

    # Calcular métricas
    accuracy_ann = accuracy_score(y_test_encoded, y_pred)
    classification_rep = classification_report(y_test_encoded, y_pred, output_dict=True)

    # Almacenar todas las métricas en un diccionario
    scores_ann = {
        'test_accuracy': accuracy_ann
    }

    # Imprimir las métricas
    print(f"Precisión ANN con HOG: {accuracy_ann:.2f}")
    print("\nReporte de ANN con HOG:")
    print(classification_rep)

    if plot == True:
        # Llamar a la función de validación para visualizar las imágenes y las predicciones
        #validacion(X_test_hog, y_test_encoded, y_pred_prob, class_names)

        evaluar_rendimiento(
            y_test_encoded,
            y_pred_prob,
            y_pred,
            "ANN"
        )

    return ann_model, scores_ann