import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

# Obtener el conjunto de datos
caras = fetch_olivetti_faces()
_, altura_img, ancho_img = caras.images.shape
print(caras.images.shape)

# Dividir el conjunto de datos
N_IDENTIDADES = len(np.unique(caras.target)) # cuántos individuos diferentes hay en el conjunto de datos
TAM_GALERIA = 8                            # utilizar las primeras TAM_GALERIA imágenes por individuo para entrenamiento, el resto para prueba

indices_galeria = []
indices_prueba = []
for i in range(N_IDENTIDADES):
    indices = list(np.where(caras.target == i)[0])
    indices_galeria += indices[:TAM_GALERIA]
    indices_prueba += indices[TAM_GALERIA:]

x_entrenamiento = caras.images[indices_galeria].reshape(-1, altura_img, ancho_img, 1) # remodelar para entrada CNN
y_entrenamiento = caras.target[indices_galeria]
x_prueba = caras.images[indices_prueba].reshape(-1, altura_img, ancho_img, 1)    # remodelar para entrada CNN
y_prueba = caras.target[indices_prueba]
print(x_entrenamiento.shape, x_prueba.shape)

# Visualizar conjuntos de imágenes
def mostrar_imagenes(imgs, num_filas, num_columnas):
    assert len(imgs) == num_filas*num_columnas

    total = None
    for i in range(num_filas):
        fila = None
        for j in range(num_columnas):
            if fila is None:
                fila = imgs[i*num_columnas+j].reshape(altura_img, ancho_img)*255.0
            else:
                fila = np.concatenate((fila, imgs[i*num_columnas+j].reshape(altura_img, ancho_img)*255.0), axis=1)
        if total is None:
            total = fila
        else:
            total = np.concatenate((total, fila), axis=0)

    f = plt.figure(figsize=(num_columnas, num_filas))
    plt.imshow(total, cmap='gray')
    plt.axis('off')
    plt.show()

print('ENTRENAMIENTO')
mostrar_imagenes(x_entrenamiento, N_IDENTIDADES, TAM_GALERIA)
print('PRUEBA')
mostrar_imagenes(x_prueba, N_IDENTIDADES, 10 - TAM_GALERIA)

# Crear un pequeño conjunto de validación a partir del conjunto de entrenamiento
x_entrenamiento, x_validacion, y_entrenamiento, y_validacion = train_test_split(x_entrenamiento, y_entrenamiento, test_size=0.1, random_state=42)

# Implementación del LDA
lda = LDA(n_components=N_IDENTIDADES - 1)
lda.fit(x_entrenamiento.reshape(x_entrenamiento.shape[0], -1), y_entrenamiento)  # Aplanar x_entrenamiento para LDA
predicciones_lda = lda.predict(x_prueba.reshape(x_prueba.shape[0], -1))  # Aplanar x_prueba para LDA

# Implementación del SVM
svm = SVC(kernel='linear')
svm.fit(x_entrenamiento.reshape(x_entrenamiento.shape[0], -1), y_entrenamiento)  # Aplanar x_entrenamiento para SVM
predicciones_svm = svm.predict(x_prueba.reshape(x_prueba.shape[0], -1))  # Aplanar x_prueba para SVM

# Crear la arquitectura de la CNN (LeNet-5)
modelo = models.Sequential()
modelo.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(altura_img, ancho_img, 1)))
modelo.add(layers.MaxPooling2D(pool_size=(2, 2)))
modelo.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
modelo.add(layers.MaxPooling2D(pool_size=(2, 2)))
modelo.add(layers.Flatten())
modelo.add(layers.Dense(120, activation='relu'))
modelo.add(layers.Dense(84, activation='relu'))
modelo.add(layers.Dense(N_IDENTIDADES, activation='softmax'))

modelo.summary()

# Compilar y entrenar la CNN
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
historial = modelo.fit(x_entrenamiento, y_entrenamiento, epochs=20, batch_size=32, validation_data=(x_validacion, y_validacion))

# Guardar el modelo con la mayor precisión en el conjunto de validación
mejor_ruta_modelo = 'mejor_modelo.h5'
mejor_precision_validacion = max(historial.history['val_accuracy'])
modelo.save(mejor_ruta_modelo + '.keras')

# Comparar el rendimiento de LDA, SVM y CNN en términos de F-Score y matriz de confusión
predicciones_cnn = modelo.predict(x_prueba)
etiquetas_predicciones_cnn = np.argmax(predicciones_cnn, axis=1)

f_score_medio_lda = f1_score(y_prueba, predicciones_lda, average='macro')
f_score_medio_svm = f1_score(y_prueba, predicciones_svm, average='macro')
f_score_medio_cnn = f1_score(y_prueba, etiquetas_predicciones_cnn, average='macro')

matriz_confusion_lda = confusion_matrix(y_prueba, predicciones_lda)
matriz_confusion_svm = confusion_matrix(y_prueba, predicciones_svm)
matriz_confusion_cnn = confusion_matrix(y_prueba, etiquetas_predicciones_cnn)

print("F-Score Medio - LDA:", f_score_medio_lda)
print("F-Score Medio - SVM:", f_score_medio_svm)
print("F-Score Medio - CNN:", f_score_medio_cnn)

print("Matriz de Confusión - LDA:")
print(matriz_confusion_lda)
print("Matriz de Confusión - SVM:")
print(matriz_confusion_svm)
print("Matriz de Confusión - CNN:")
print(matriz_confusion_cnn)
