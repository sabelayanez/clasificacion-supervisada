from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

def build_mobilenetv2_model(input_shape=(256, 256, 3), num_classes=5, dense_units=1024, dropout_rate=0.5, learning_rate=0.001):
    # Cargar MobileNetV2 preentrenado sin la capa superior
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Congelar las capas convolucionales para no entrenarlas inicialmente
    for layer in base_model.layers:
        layer.trainable = False

    # Construir el modelo con la capa GlobalAveragePooling2D y las capas densas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reducción de la dimensionalidad
    x = Dense(dense_units, activation='relu')(x)  # Capa densa con hiperparámetro ajustable
    x = Dropout(dropout_rate)(x)  # Capa de Dropout para regularización
    predictions = Dense(num_classes, activation='softmax')(x)  # Capa de salida con softmax

    # Crear el modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_mobilenetv2_model(X_train, y_train, X_test, y_test, input_shape=(256, 256, 3), num_classes=5, epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001):

    model = build_mobilenetv2_model(input_shape=input_shape, num_classes=num_classes, dense_units=dense_units, dropout_rate=dropout_rate, learning_rate=learning_rate)
    
    # Configurar EarlyStopping para evitar sobreajuste
    earlystop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlystop_callback],
        verbose=1
    )
        # Evaluar el modelo en los datos de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en los datos de prueba: {accuracy:.2f}")
    
    # Visualización del historial de entrenamiento
    plt.figure(figsize=(12, 6))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.show()
    
    return model

from tensorflow.keras.utils import to_categorical

num_classes = len(np.unique(y_train_encoded))  # Asegurar que tenemos el número correcto de clases
y_train_onehot = to_categorical(y_train_encoded, num_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes)

# Llamada a la función de entrenamiento
model_1 = train_mobilenetv2_model(X_train_rgb, y_train_onehot, X_test_rgb, y_test_onehot, epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001)
model_2 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3), epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.001)
model_3 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3),epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.5, learning_rate=0.0001)
model_4 = train_mobilenetv2_model(X_train_rgb_64, y_train_onehot, X_test_rgb_64, y_test_onehot, input_shape=(64, 64, 3),epochs=10, batch_size=32, dense_units=1024, dropout_rate=0.3, learning_rate=0.001)