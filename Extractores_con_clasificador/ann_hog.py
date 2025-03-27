from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from utils import validacion, extract_hog_features
from constants import epochs

def ann_hog(X_train, y_train, X_test, y_test, num_classes, class_names):
    # Convertir las etiquetas a one-hot encoding
    
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)
    
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)

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
    y_pred_ann = ann_model.predict(X_test_hog)
    y_pred_ann_classes = np.argmax(y_pred_ann, axis=1)  # Obtener clases predichas

    # Calcular métricas
    accuracy_ann = accuracy_score(y_test, y_pred_ann_classes)
    classification_rep = classification_report(y_test, y_pred_ann_classes, output_dict=True)
    
    # Calcular ROC AUC si es una clasificación multiclase
    try:
        roc_auc = roc_auc_score(y_test_onehot, y_pred_ann, multi_class='ovr')
    except ValueError:
        roc_auc = None  # No se puede calcular si hay solo una clase o problemas con la métrica

    # Almacenar todas las métricas en un diccionario
    scores_ann = {
        'accuracy': accuracy_ann,
        'classification_report': classification_rep,
        'roc_auc': roc_auc
    }

    # Imprimir las métricas
    print(f"Precisión ANN con HOG: {accuracy_ann:.2f}")
    print("\nReporte de ANN con HOG:")
    print(classification_rep)

    # Llamar a la función de validación para visualizar las imágenes y las predicciones
    #validacion(X_test_hog, y_test, y_pred_ann, class_names)

    return ann_model, scores_ann