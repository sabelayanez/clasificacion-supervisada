import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

import kagglehub

from constants import dataset_name, opciones, CV, scoring
from utils import cargar_imagenes, evaluar_rendimiento, save_excel_cv

from modelos.RegresionLogistica import regresion_logistica
from modelos.CNN import cnn1, cnn2
from modelos.KNN import knn
from modelos.ArbolDeDecision import arbol_decision

from tensorflow.keras.utils import to_categorical
from modelos.MobileNetV2 import train_mobilenetv2_model

import matplotlib.pyplot as plt

def load_model():
    print("Seleccione una opción:")
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}. {opcion}")
    seleccion = int(input("Ingrese el número de la opción: "))
    
    # Validar que la selección sea válida
    if 1 <= seleccion <= len(opciones):
        modelo = opciones[seleccion - 1]
    else:
        print("Selección no válida.")
    # Descargar dataset
    path = kagglehub.dataset_download(dataset_name)

    # cargar train y test
    X_train_rgb, y_train = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/train'), channel_mode="rgb")
    X_test_rgb, y_test = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/test'), channel_mode="rgb")
    X_train_gray, _ = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/train'), channel_mode="grayscale")
    X_test_gray, _ = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/test'), channel_mode="grayscale")
    X_train_rgb_64, _ = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/train'), target_size=(64, 64))
    X_test_rgb_64, _ = cargar_imagenes(os.path.join(path, 'flowers/flowers/flower_photos/test'), target_size=(64, 64))

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    epochs = 100
    batch_size = 100
    history = []

    if modelo == "regresion_logistica_rgb":
        ## si es regresión logística rgb ##
        modelLR_rgb, scoresLR_rgb = regresion_logistica(X_train_rgb, y_train_encoded)
        save_excel_cv(scoresLR_rgb, "Regresión Logística RGB")
    elif modelo == "regresion_logistica_gray":
        ## si es regresión logística gray ##
        modelLR_gray, scoresLR_gray = regresion_logistica(X_train_gray, y_train_encoded, X_test_gray, y_test_encoded)
        save_excel_cv(scoresLR_gray, "Regresión Logística gray")

    elif modelo == "cnn_opcion_1":
        ## si es CNN1 ##
        model1 = cnn1()
        model1.summary()
        
        history.append(model1.fit(X_train_rgb_64, y_train_encoded,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test_rgb_64, y_test_encoded)))
        
    elif modelo == "cnn_opcion_2":

        ## si es CNN2 ##    
        model2 = cnn2()
        model2.summary()

        history.append(model2.fit(X_train_rgb_64, y_train_encoded,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_rgb_64, y_test_encoded)))
    
    elif modelo == "knn":
        ## si es KNN ## 
        modelKNN, pcaKNN, scoresKNN = knn(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded)
        save_excel_cv(scoresKNN, "KNN")

    elif modelo == "arbol_de_decision":
        model_tree = arbol_decision(X_train_rgb_64, y_train_encoded)
        # Evaluar el modelo con las funciones definidas previamente
        evaluar_rendimiento(model_tree, X_test_rgb_64, y_test, "Árbol de Decisión")

    elif modelo == "":
        num_classes = len(np.unique(y_train_encoded))  # Asegurar que tenemos el número correcto de clases
        y_train_onehot = to_categorical(y_train_encoded, num_classes)
        y_test_onehot = to_categorical(y_test_encoded, num_classes)

        # Llamada a la función de entrenamiento
        model_1 = train_mobilenetv2_model(X_train_rgb, y_train_onehot, X_test_rgb, y_test_onehot, epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001)
        model_2 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3), epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001)
        model_3 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3),epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.0001)
        model_4 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3),epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.3, learning_rate=0.001)

    else: 
        print("Se realizarán todos los modelos")  
        ## regresión logística rgb ##
        modelLR_rgb, scoresLR_rgb = regresion_logistica(X_train_rgb, y_train_encoded)
        save_excel_cv(scoresLR_rgb, "Regresión Logística RGB")
        ## regresión logística gray ##
        modelLR_gray, scoresLR_gray = regresion_logistica(X_train_gray, y_train_encoded, X_test_gray, y_test_encoded)
        save_excel_cv(scoresLR_gray, "Regresión Logística gray")
        ## CNN1 ##
        model1 = cnn1()
        model1.summary()
        
        history.append(model1.fit(X_train_rgb_64, y_train_encoded,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test_rgb_64, y_test_encoded)))
        
        ## CNN2 ##    
        model2 = cnn2()
        model2.summary()

        history.append(model2.fit(X_train_rgb_64, y_train_encoded,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_rgb_64, y_test_encoded)))
    
        ## KNN ## 
        modelKNN, pcaKNN, scoresKNN = knn(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded)

        save_excel_cv(scoresKNN, "KNN")

        ## Árbol de Decisión ##
        model_tree = arbol_decision(X_train_rgb_64, y_train_encoded)

        ## diagrama de cajas validación cru
        data = [scoresLR_rgb['test_accuracy'], scoresLR_gray['test_accuracy'], scoresKNN['test_accuracy']]
        _, ax = plt.subplots()
        ax.set_title('Modelos')
        ax.boxplot(data,labels=['LR_RGB','LR_GRAY','KNN'])

if __name__ == "__main__":
    load_model()