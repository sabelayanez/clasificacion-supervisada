from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler

from skimage.transform import resize
from skimage.color import rgb2gray

from PIL import Image

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import random
from skimage.feature import hog
from constants import excel_filename, excel_filename_cv, scoring, class_names

## funciones para validación cruzada ##
def save_excel_cv(scores, nombre_metodo):
    datos = {
        'precision': [np.mean(scores['test_precision'])],
        'recall': [np.mean(scores['test_recall'])], 
        'f1': [np.mean(scores['test_f1'])], 
        'accuracy': [np.mean(scores['test_accuracy'])], 
        'roc_auc': [np.mean(scores['test_roc_auc'])]
    }
    if os.path.exists(excel_filename_cv):
        results = pd.read_excel(excel_filename_cv, index_col=0)
        df = pd.DataFrame(datos, index=[nombre_metodo])
        results = pd.concat([results, df], ignore_index=False)
        results.to_excel(excel_filename_cv)
    else:
        results = pd.DataFrame(datos)
        results.index = [nombre_metodo]
        results.to_excel(excel_filename_cv)

def evaluar_mobilenetv2(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)  # Predicciones de probabilidades
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convertir las probabilidades en clases

    y_test_class = np.argmax(y_test, axis=1)  

    # Calcular Accuracy
    accuracy = accuracy_score(y_test_class, y_pred)
    
    # Calcular Precision
    precision = precision_score(y_test_class, y_pred, average='macro', zero_division=0)
    
    # Calcular Recall
    recall = recall_score(y_test_class, y_pred, average='macro', zero_division=0)
    
    # Calcular F1 Score
    f1 = f1_score(y_test_class, y_pred, average='macro', zero_division=0)
    
    # Calcular ROC AUC Score (solo para clasificación binaria o multiclasificación)
    # Si es clasificación binaria, usar average='macro', para multiclase 'ovr' (One-vs-Rest)
    if y_test_class.shape[0] > 1:
        roc_auc = roc_auc_score(y_test_class, y_pred_prob, multi_class='ovr', average='macro')
    else:
        roc_auc = roc_auc_score(y_test_class, y_pred_prob[:, 1])  # Para clasificación binaria

    # Mostrar las métricas
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    datos = {
        'test_precision': [precision],
        'test_recall': [recall], 
        'test_f1': [f1], 
        'test_accuracy': [accuracy], 
        'test_roc_auc': [roc_auc]
    }

    save_excel_cv(datos, "MobileNetV2")
    
## guardar datos en excel ##
def save_to_excel(datos):
    # Si el archivo ya existe, se leerá y se agregará nueva información
    metodo =  datos.pop("Método")
    if os.path.exists(excel_filename):
        results = pd.read_excel(excel_filename, index_col=0)
        df = pd.DataFrame(datos, index=metodo)
        results = pd.concat([results, df], ignore_index=False)
        results.to_excel(excel_filename)
    else:
        results = pd.DataFrame(datos)
        results.index = metodo
        results.to_excel(excel_filename)
        
## graficar métricas desde excel ##
def graficar_metrics_desde_excel(file_path):
    # Leer el archivo Excel
    df = pd.read_excel(file_path, index_col=0)

    if not all(col in df.columns for col in scoring.keys()):
        raise ValueError(f"El archivo debe contener las siguientes columnas: {scoring.keys()}")

    # Extraer los modelos (que están en el índice)
    modelos = df.index
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


    # Graficar la media de cada métrica para cada modelo
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(modelos, df[metric], color='g')
        plt.title(f'Media de {metric.capitalize()} para cada Modelo')
        print(metric)
        print(df[metric])
        plt.xlabel('Modelo')
        plt.ylabel(f'Media de {metric.capitalize()}')
        plt.ylim([0, 1])  # Asegurarse de que el rango de la métrica esté entre 0 y 1
        plt.xticks(rotation=45, ha='right')
        plt.show()


#función para hacer la gráfica después de evaluar el rendimiento
def plot_rendimiento(matriz_confusion, clases, fpr_micro, tpr_micro, roc_auc_micro, fpr, tpr, roc_auc, nombre_metodo):
    #matriz de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=clases)
    disp.plot()
    disp.figure_.suptitle(f"Matriz de confusión para {nombre_metodo}")
    disp.figure_.set_dpi(100)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.show()

    #roc y auc
    plt.figure()
    plt.plot(fpr_micro, tpr_micro, color='red', lw=2, label='Curva ROC micro-average (AUC = %0.3f)' % roc_auc_micro)
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f"Curva ROC para {nombre_metodo}")
    plt.legend(loc="lower right")
    plt.show()

    # Curvas ROC por clase
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'blue', 'violet', 'gold', 'orange', 'pink', 'tan', 'purple', 'lime', 'red']
    for i in range(len(clases)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=1, label='ROC clase %i (area = %0.3f)' % (i, roc_auc[i]))

    plt.plot(fpr_micro, tpr_micro, color='red', lw=2, linestyle=':', label='Curva ROC micro-average (AUC = %0.3f)' % roc_auc_micro)
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f"Curva ROC por clase para {nombre_metodo}")
    plt.legend(loc="lower right")
    plt.show()

## evaluar rendimiento: matriz de confusión y curva ROC_AUC ##
def evaluar_rendimiento(y_test_encoded, y_pred_prob, y_pred, nombre_metodo):
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    y_val_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(y_val_bin.shape[1]):  # Para cada clase
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ROC micro (promedio de todas las clases)
    fpr_micro, tpr_micro, _ = roc_curve(y_val_bin.ravel(), y_pred_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plot_rendimiento(
        cm,
        class_names,
        fpr_micro,
        tpr_micro,
        roc_auc_micro,
        fpr,
        tpr,
        roc_auc,
        nombre_metodo
    )


## cargar imágenes ##
def cargar_imagenes(image_path, target_size=(256, 256), channel_mode="rgb"):

    img_list = []
    labels = []
    classes = os.listdir(image_path)

    for folder in classes:
        folder_path = os.path.join(image_path, folder)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img = Image.open(os.path.join(folder_path, filename)).convert("RGB")
                    img_array = np.array(img)

                    # Redimensionar
                    img_resized = resize(img_array, target_size, anti_aliasing=True)

                    # Modos de canal
                    if channel_mode == "grayscale":
                        img_resized = rgb2gray(img_resized)  # Convertir a escala de grises
                    elif channel_mode == "r":  # Canal rojo
                        img_resized = img_resized[:, :, 0]
                    elif channel_mode == "g":  # Canal verde
                        img_resized = img_resized[:, :, 1]
                    elif channel_mode == "b":  # Canal azul
                        img_resized = img_resized[:, :, 2]
                    elif channel_mode == "rgb":
                        img_resized = (img_resized * 255).astype(np.uint8)  # Restaurar valores de píxeles

                    img_resized = img_resized / 255.0  # Normalizar
                    img_list.append(img_resized)
                    labels.append(folder)  # Guardar etiqueta

    return np.array(img_list), np.array(labels)

def graficar_metrics_desde_excel(file_path):
    # Leer el archivo Excel
    df = pd.read_excel(file_path, index_col=0)

    if not all(col in df.columns for col in scoring.keys()):
        raise ValueError(f"El archivo debe contener las siguientes columnas: {scoring.keys()}")

    # Extraer los modelos (que están en el índice)
    modelos = df.index
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


    # Graficar la media de cada métrica para cada modelo
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(modelos, df[metric], color='g')
        plt.title(f'Media de {metric.capitalize()} para cada Modelo')
        plt.xlabel('Modelo')
        plt.ylabel(f'Media de {metric.capitalize()}')
        plt.ylim([0, 1])  # Asegurarse de que el rango de la métrica esté entre 0 y 1
        plt.xticks(rotation=45, ha='right')
        plt.show()

def validacion(X_test, y_test_encoded, y_pred, class_names):
    # Visualizar las imágenes y las predicciones
    plt.figure(figsize=(12, 6))  # Ajustar tamaño

    for i in range(10):  # Mostrar 10 imágenes
        index = random.randint(0, X_test.shape[0] - 1)
        image_to_show = X_test[index]
        true_label_num = y_test_encoded[index]  # Etiqueta real codificada
        
        # Obtener la predicción
        pred_label_num = np.argmax(y_pred[index])  # Etiqueta predicha
        
        true_label_str = class_names[true_label_num]  # Nombre de la clase real
        pred_label_str = class_names[pred_label_num]  # Nombre de la clase predicha

        # Dibujar imagen
        plt.subplot(2, 5, i + 1)
        plt.imshow(image_to_show)
        plt.axis('off')

        # Mostrar la predicción
        if true_label_num == pred_label_num:
            plt.title(pred_label_str)
        else:
            plt.title(f"{pred_label_str} != {true_label_str}", color='red')

def extract_hog_features(images):
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feature)
    return np.array(hog_features)