from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

excel_filename = "results.xlsx"
excel_filename_cv = "results_cv.xlsx"
dataset_name = "sujaykapadnis/flowers-image-classification"
opciones = [
    "regresion_logistica_rgb", 
    "regresion_logistica_gray", 
    "cnn_opcion_1", 
    "cnn_opcion_2", 
    "knn", 
    "arbol_de_decision", 
    "arbol_de_decision_vgg", 
    "arbol_de_decision_vgg_pca",
    "mobile_net_v2",
    "random_forest",
    "rforest_vgg16_pca",
    "rforest_vgg16_pca_hog",
    "hog"
    ]
## cross validation
CV = 5

epochs = 100
batch_size = 100

# exactitud, sensibilidad, especificidad, precisión, f1-score y área bajo la curva (AUC)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro',
    'roc_auc': 'roc_auc_ovr'
}

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']