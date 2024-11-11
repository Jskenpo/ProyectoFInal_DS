# Proyecto Final Data Science

## Descripción

Este proyecto tiene como objetivo predecir el comportamiento de los usuarios en una plataforma de comercio electrónico utilizando cuatro modelos de machine learning diferentes: **SVM (Support Vector Machine)**, **LSTM (Long Short-Term Memory)**, **PCA (Principal Component Analysis)**, y **SARIMA (Seasonal Autoregressive Integrated Moving Average)**. Además, se implementó una API con Flask para realizar predicciones a través de solicitudes HTTP.

## Modelos Utilizados

### 1. **SVM (Support Vector Machine)**
   - **Objetivo**: Utilizado para clasificar si un usuario realizará una compra o no (evento de tipo `purchase` o `no purchase`).
   - **Preprocesamiento**: 
     - Normalización de la variable `price`.
     - Aplicación de One-Hot Encoding a las variables `brand` y `category`.
     - Balanceo de clases mediante el uso de SMOTE.
   - **Resultado**: El modelo SVM mostró ser el mejor rendimiento en comparación con los otros modelos en términos de precisión y generalización.

### 2. **LSTM (Long Short-Term Memory)**
   - **Objetivo**: Modelar las series temporales y predecir la compra basada en la secuencia de eventos previos.
   - **Preprocesamiento**: Los datos fueron reorganizados para capturar dependencias temporales y estructurarlos en secuencias.
   - **Resultado**: Aunque LSTM capturó la dinámica temporal, su desempeño fue inferior al de SVM en términos de precisión y capacidad de generalización.

### 3. **PCA (Principal Component Analysis)**
   - **Objetivo**: Reducir la dimensionalidad de los datos para mejorar la eficiencia del modelo, manteniendo la mayor varianza posible.
   - **Preprocesamiento**: Aplicación de PCA para reducir las características y mantener solo las componentes principales.
   - **Resultado**: PCA ayudó a acelerar el entrenamiento, pero no mejoró significativamente la predicción en comparación con los otros modelos.

### 4. **SARIMA (Seasonal Autoregressive Integrated Moving Average)**
   - **Objetivo**: Modelar los patrones estacionales en los datos de compra.
   - **Preprocesamiento**: Análisis de la estacionalidad de los datos y ajuste de parámetros SARIMA para captar la dinámica temporal.
   - **Resultado**: Aunque SARIMA es efectivo para modelar datos temporales, no superó al modelo SVM en precisión y eficiencia general.

## API para Predicciones

### Descripción
Se implementó una API usando Flask que permite hacer predicciones sobre el comportamiento de los usuarios en base a los datos de entrada. La API toma un JSON con las variables `price`, `brand`, y `category`, realiza el preprocesamiento necesario, y devuelve la predicción del modelo SVM.

### Funcionalidad
- **Endpoint**: `/predict`
- **Método HTTP**: `POST`
- **Entrada**: JSON con las siguientes claves:
  - `price`: Precio del producto.
  - `brand`: Marca del producto.
  - `category`: Categoría del producto.
  
  Ejemplo de entrada:
  ```json
  {
    "price": 15,
    "brand": "Samsung",
    "category": "electronics.smartphone"
  }
  ```

- **Proceso**:
  1. Normalización de la variable `price`.
  2. Aplicación de One-Hot Encoding para las variables `brand` y `category`.
  3. Uso del modelo SVM para hacer la predicción.
  
- **Salida**: JSON con la predicción:
  ```json
  {
    "predictions": [1]
  }
  ```
  Donde `1` indica una compra (`purchase`), y `0` indica no compra (`no purchase`).

### Requisitos
- Python 3.x
- Flask
- scikit-learn
- joblib
- pandas

## Cómo Ejecutar

### 1. **Instalar Dependencias**
   Asegúrese de tener todas las dependencias necesarias instaladas:
   

### 2. **Ejecutar el Servidor Flask**
   Para iniciar la API, ejecute el siguiente comando:
   ```bash
   python API_Prod.py
   ```
   Esto pondrá en marcha el servidor en `http://127.0.0.1:5000`.

### 3. **Realizar Predicciones**
   Puede realizar predicciones enviando un `POST` al endpoint `/predict` con un cuerpo JSON con los datos del producto.

## Conclusión

Este proyecto implementa y compara cuatro enfoques de modelado para predecir el comportamiento de compra de los usuarios. A través de la API, se puede acceder fácilmente a las predicciones del modelo SVM, el cual resultó ser el modelo más efectivo para esta tarea.

