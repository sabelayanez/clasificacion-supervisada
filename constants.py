excel_filename = "results.xlsx"
dataset_name = "sujaykapadnis/flowers-image-classification"
opciones = ["regresion_logistica_rgb", "regresion_logistica_gray", "cnn_opcion_1", "cnn_opcion_2", "knn"]
## cross validation
CV = 20
scoring = ['precision_macro', 'recall_macro', 'precision_micro', 'recall_micro', 'f1_macro', 'accuracy', 'roc_auc']