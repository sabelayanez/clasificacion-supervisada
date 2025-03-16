from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

def regresion_logistica(X_train, y_train_encoded, X_test, y_test_encoded):
    # Aplanar las imágenes de 4D a 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Aplanar a 2D: [n_samples, n_features]
    #X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Aplanar a 2D: [n_samples, n_features]

    # Entrenar modelo de Regresión Logística
    modelLR = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    modelLR.fit(X_train_flat, y_train_encoded)

    # Hacer predicciones
    #y_train_pred = modelLR.predict(X_train_flat)
    #y_test_pred = modelLR.predict(X_test_flat)

    # Calcular métricas de evaluación
    #accuracy_train = accuracy_score(y_train_encoded, y_train_pred)
    #accuracy_test = accuracy_score(y_test_encoded, y_test_pred)

    # Imprimir resultados
    #print(f"Precisión en entrenamiento: {accuracy_train:.2f}")
    #print(f"Precisión en test: {accuracy_test:.2f}")

    # Número de aciertos
    #aciertos_train = sum(y_train_encoded == y_train_pred)
    #aciertos_test = sum(y_test_encoded == y_test_pred)
    #print(f"Número de aciertos en entrenamiento: {aciertos_train} de {len(y_train_pred)}")
    #print(f"Número de aciertos en test: {aciertos_test} de {len(y_test_pred)}")

    # Visualización de una imagen de prueba
    #img_test = X_test[14]  # Obtener la imagen
    #if img_test.ndim == 3 and img_test.shape[-1] == 3:  # Si es RGB, convertir a escala de grises
     #   img_test = np.dot(img_test[..., :3], [0.2989, 0.5870, 0.1140])  # Conversión a escala de grises

    #img_test = img_test.squeeze()  # Eliminar dimensiones extra
    #plt.imshow(img_test, cmap='gray')
    #plt.title(f"Etiqueta real: {label_encoder.inverse_transform([y_test_encoded[0]])[0]}\n"
     #         f"Predicción: {label_encoder.inverse_transform([y_test_pred[0]])[0]}")
    #plt.axis("off")
    #plt.show()

    return modelLR