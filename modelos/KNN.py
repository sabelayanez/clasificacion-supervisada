import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def knn(X_train_rgb, y_train_encoded, X_test_rgb, y_test_encoded):
    X_train_flattened = X_train_rgb.reshape(X_train_rgb.shape[0], -1)
    X_test_flattened = X_test_rgb.reshape(X_test_rgb.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flattened)
    X_test_scaled = scaler.transform(X_test_flattened)
    
    pca = PCA(n_components=50)  # You can adjust n_components as needed
    X_train_pca = pca.fit_transform(X_train_scaled)
    #X_test_pca = pca.transform(X_test_scaled)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_pca, y_train_encoded)
    
    # Return both the trained KNN model and PCA model as a tuple
    return knn_model, pca

    # Aplanar las imágenes
    #X_train_flat = X_train.reshape(X_train.shape[0], -1)
    #X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Normalizar los datos
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train_flat)
    #X_test_scaled = scaler.transform(X_test_flat)

    # Reducir dimensionalidad con PCA
    #pca = PCA(n_components=n_components)
    #X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)

    # Entrenar modelo KNN con mejor k
    #model = KNeighborsClassifier(n_neighbors=k)
    #model.fit(X_train_pca, y_train)

    # Predicciones
    # y_train_pred = model.predict(X_train_pca)
    # y_test_pred = model.predict(X_test_pca)

    # Evaluación
    # accuracy_train = accuracy_score(y_train, y_train_pred)
    # accuracy_test = accuracy_score(y_test, y_test_pred)

    # Imprimir resultados
    # print(f"Precisión en entrenamiento: {accuracy_train:.2f}")
    # print(f"Precisión en test: {accuracy_test:.2f}")

    # Número de aciertos
    # aciertos_train = sum(y_train_encoded == y_train_pred)
    # aciertos_test = sum(y_test_encoded == y_test_pred)
    # print(f"Número de aciertos en entrenamiento: {aciertos_train} de {len(y_train_pred)}")
    # print(f"Número de aciertos en test: {aciertos_test} de {len(y_test_pred)}")

    # Visualización de una imagen de prueba
    # img_test = X_test[14]  # Obtener la imagen
    # if img_test.ndim == 3 and img_test.shape[-1] == 3:  # Si es RGB, convertir a escala de grises
      #  img_test = np.dot(img_test[..., :3], [0.2989, 0.5870, 0.1140])  # Conversión a escala de grises

    # img_test = img_test.squeeze()  # Eliminar dimensiones extra
    # plt.imshow(img_test, cmap='gray')
    # plt.title(f"Etiqueta real: {label_encoder.inverse_transform([y_test_encoded[14]])[0]}\n"
      #        f"Predicción: {label_encoder.inverse_transform([y_test_pred[14]])[0]}")
    # plt.axis("off")
    # plt.show()

    # Guardar distancias y predicciones en un archivo Excel
    # save_to_excel(accuracy_train, accuracy_test, aciertos_train, aciertos_test, len(y_train_pred), len(y_test_pred), filename="knn_results.xlsx")

