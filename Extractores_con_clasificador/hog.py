def main_hog(X_train, X_test, y_train, y_test, num_classes):
    # Extraer características HOG
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # Entrenar y evaluar ANN con características HOG
    print("Entrenando y evaluando ANN con características HOG...")
    ann_model = train_evaluate_ann_with_hog(X_train_hog, y_train, X_test_hog, y_test, num_classes)

    # Entrenar y evaluar SVM con características HOG
    print("\nEntrenando y evaluando SVM con características HOG...")
    svm_model = train_evaluate_svm_with_hog(X_train_hog, y_train, X_test_hog, y_test)

    return ann_model, svm_model

def extract_hog_features(images):
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feature)
    return np.array(hog_features)