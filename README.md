# Reconocimiento de Caras con LDA, SVM y CNN
Este es un programa que utiliza algoritmos de reconocimiento de caras como LDA (Análisis Discriminante Lineal), SVM (Máquinas de Soporte Vectorial) y CNN (Redes Neuronales Convolucionales) para reconocer rostros en el conjunto de datos Olivetti Faces.

# Autor:
José Luis Flores Tito - Analista de Ciberseguridad

## Descripción
El programa utiliza el conjunto de datos Olivetti Faces para entrenar y evaluar diferentes modelos de reconocimiento facial. Primero, se divide el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba. Luego, se implementan tres algoritmos diferentes:

1. **LDA (Análisis Discriminante Lineal)**: Se aplica el algoritmo de LDA para reducir la dimensionalidad de las imágenes y luego se realiza la clasificación mediante una regresión logística.

2. **SVM (Máquinas de Soporte Vectorial)**: Se entrena un modelo SVM lineal para la clasificación de las imágenes.

3. **CNN (Redes Neuronales Convolucionales)**: Se crea una arquitectura de red neuronal convolucional basada en LeNet-5 y se entrena para clasificar las imágenes.

Finalmente, se compara el rendimiento de los tres algoritmos en términos de F-Score y se muestra la matriz de confusión para cada uno.

## Requisitos
- Python 3.x
- Bibliotecas de Python: matplotlib, numpy, tensorflow, scikit-learn

## Cómo usar
Asegúrate de tener las bibliotecas requeridas instaladas. Puedes instalarlas usando pip:

```bash
pip install matplotlib numpy tensorflow scikit-learn
```
Ejecuta el programa:
```bash
python reconocimiento_caras.py
```
## Resultados
El programa mostrará las imágenes de entrenamiento y prueba, junto con los resultados del reconocimiento facial utilizando los algoritmos LDA, SVM y CNN. También se mostrará el F-Score promedio y la matriz de confusión para cada algoritmo.

## Contactos:
Si te gusta mi trabajo o estás buscando consultoría para tus proyectos, Pentesting, servicios de RED TEAM - BLUE TEAM, implementación de normas de seguridad e ISOs, controles IDS - IPS, gestión de SIEM, implementación de topologías de red seguras, entrenamiento e implementación de modelos de IA, desarrollo de sistemas, Apps Móviles, Diseño Gráfico, Marketing Digital y todo lo relacionado con la tecnología, no dudes en contactarme al +591 75764248 y con gusto trabajare contigo.
