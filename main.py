import os
import numpy as np

from sklearn.preprocessing import LabelEncoder

import kagglehub

from constants import dataset_name, opciones, CV, scoring, class_names
from utils import cargar_imagenes, evaluar_rendimiento, save_excel_cv, evaluar_mobilenetv2

from modelos.RegresionLogistica import regresion_logistica
from modelos.CNN import cnn_cross_validation;
from modelos.KNN import knn, knn_with_gridsearch
from modelos.ArbolDeDecision import arbol_decision, arbol_decision_vgg16, arbol_vgg16_pca
from modelos.MobileNetV2 import train_mobilenetv2_model
from modelos.RandomForest import random_forest, rforest_vgg16_pca, rforest_vgg16_pca_hog

from tensorflow.keras.utils import to_categorical
from modelos.MobileNetV2 import train_mobilenetv2_model

import matplotlib.pyplot as plt

def load_model():
    print("Seleccione una opción:")
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}. {opcion}")
    seleccion = int(input("Ingrese el número de la opción: "))
    
    # Validar que la selección sea válida
    modelo = opciones[seleccion - 1]
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
        modelLR_rgb, scoresLR_rgb = regresion_logistica(X_train_rgb, y_train_encoded, X_train_rgb, y_test_encoded)
        save_excel_cv(scoresLR_rgb, "Regresión Logística RGB")

    elif modelo == "regresion_logistica_gray":
        ## si es regresión logística gray ##
        modelLR_gray, scoresLR_gray = regresion_logistica(X_train_gray, y_train_encoded, X_test_gray, y_test_encoded)
        save_excel_cv(scoresLR_gray, "Regresión Logística gray")

    elif modelo == "cnn_opcion_1":
        ## si es CNN1 ##
        cnn_cross_validation(X_train_rgb_64, y_train_encoded, 'cnn1', X_test_rgb_64, y_test_encoded)
        
    elif modelo == "cnn_opcion_2":
        ## si es CNN2 ##    
        cnn_cross_validation(X_train_rgb_64, y_train_encoded, 'cnn2', X_test_rgb_64, y_test_encoded)

    elif modelo == "knn":
        ## si es KNN ## 
        modelKNN, pcaKNN, scoresKNN = knn_with_gridsearch(X_train_rgb_64, y_train_encoded, X_test_rgb_64, y_test_encoded)
        save_excel_cv(scoresKNN, "KNN")

    elif modelo == "arbol_de_decision":
        model_tree, grid_search = arbol_decision(X_train_rgb_64, y_train_encoded)
        # Evaluar el modelo con las funciones definidas previamente
        #evaluar_rendimiento(model_tree, X_test_rgb_64, y_test, "Árbol de Decisión")
    
    elif modelo == "arbol_de_decision_vgg":
        model_tree, scores_tree = arbol_decision_vgg16(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded)
        # Evaluar el modelo con las funciones definidas previamente
        save_excel_cv(scores_tree, "Árbol de Decisión VGG16")
    elif modelo == "arbol_de_decision_vgg_pca":
        model_tree, scores_tree = arbol_vgg16_pca(X_train_rgb_64, y_train_encoded)
        # Evaluar el modelo con las funciones definidas previamente
        save_excel_cv(scores_tree, "Árbol de Decisión VGG16 PCA")
    elif modelo == "mobile_net_v2":
        num_classes = len(np.unique(y_train_encoded))  # Asegurar que tenemos el número correcto de clases

        y_train_onehot = to_categorical(y_train_encoded, num_classes)
        y_test_onehot = to_categorical(y_test_encoded, num_classes)

        # Llamada a la función de entrenamiento
        modelMobilenetv2 = train_mobilenetv2_model(X_train_rgb, y_train_onehot, X_test_rgb, y_test_onehot, y_test_encoded, epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001)
        # Suponiendo que ya tienes el modelo entrenado y los datos de prueba listos:
        evaluar_mobilenetv2(modelMobilenetv2, X_test_rgb, y_test_onehot)

    elif modelo == "random_forest":
        modelRF, scoresRF = random_forest(X_train_rgb_64, y_train_encoded, X_test_rgb_64, y_test_encoded, class_names)
        save_excel_cv(scoresRF, "RANDOM FOREST")
    
    elif modelo == "rforest_vgg16_pca":
        modelRF_vgg_pca, scoresRF_vgg_pca = rforest_vgg16_pca(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded)        
        save_excel_cv(scoresRF, "RANDOM FOREST VGG16 PCA")
    
    elif modelo == "rforest_vgg16_pca_hog":
        modelRF_hog, scoresRF_hog = rforest_vgg16_pca_hog(X_train_rgb_64, y_train_encoded, X_test_rgb_64, y_test_encoded, X_train_gray, X_test_gray)

    else: 
        print("Se realizarán todos los modelos")  
        ## regresión logística rgb ##
        _, scoresLR_rgb = regresion_logistica(X_train_rgb, y_train_encoded)
        save_excel_cv(scoresLR_rgb, "Regresión Logística RGB")
        ## regresión logística gray ##
        _, scoresLR_gray = regresion_logistica(X_train_gray, y_train_encoded, X_test_gray, y_test_encoded)
        save_excel_cv(scoresLR_gray, "Regresión Logística gray")
        ## CNN1 ##
        cnn_cross_validation(X_train_rgb_64, y_train_encoded, 'cnn1', X_test_rgb_64, y_test_encoded)
        
        ## CNN2 ##    
        cnn_cross_validation(X_train_rgb_64, y_train_encoded, 'cnn2', X_test_rgb_64, y_test_encoded)
    
        ## KNN ## 
        _, _, scoresKNN = knn_with_gridsearch(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded)
        save_excel_cv(scoresKNN, "KNN")

        ## Árbol de Decisión ##
        model_tree = arbol_decision(X_train_rgb_64, y_train_encoded)
        evaluar_rendimiento(model_tree, X_test_rgb_64, y_test, "Árbol de Decisión")

        ## Árbol de Decisión VGG16 ##
        model_tree = arbol_decision_vgg16(X_train_rgb_64, y_train_encoded)
        evaluar_rendimiento(model_tree, X_test_rgb_64, y_test, "Árbol de Decisión VGG16")

        ## diagrama de cajas validación cru
        data = [scoresLR_rgb['test_accuracy'], scoresLR_gray['test_accuracy'], scoresKNN['test_accuracy']]
        _, ax = plt.subplots()
        ax.set_title('Modelos')
        ax.boxplot(data,labels=['LR_RGB','LR_GRAY','KNN'])

if __name__ == "__main__":
    load_model()