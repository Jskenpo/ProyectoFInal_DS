# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
"""
# Limpieza de datos 
"""

# %%
df = pd.read_csv('sampled_data_25k.csv')

df.head()

# %%
#verificar cantidad de valores nulos 

nulos = df.isnull().sum()
print('la cantidad de valores nulos es de:\n',nulos)

#verificar cantidad de valores duplicados

duplicados = df.duplicated().sum()
print('la cantidad de valores duplicados es de: ',duplicados)

# %%
# Rellenar los nulos en 'category_code' y 'brand' con 'desconocido'
df['category_code'].fillna('desconocido', inplace=True)
df['brand'].fillna('desconocido', inplace=True)

# Verificar los cambios
print(df.isnull().sum())

# %%
# Convertir la columna 'event_time' a formato datetime
df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

# Verificar si la conversión fue exitosa
print(df['event_time'].head())


# %%
"""
# Modelo de Redes Neuronales
"""

# %%

# Conteo de eventos por hora
plt.figure(figsize=(10, 6))
sns.countplot(x='hour', hue='event_type', data=df, order=sorted(df['hour'].unique()))
plt.title('Distribución de eventos por hora')
plt.xlabel('Hora')
plt.ylabel('Cantidad de eventos')
plt.legend(title='Tipo de evento')
plt.show()

# %%
# Filtrar solo eventos de 'purchase' y 'cart'
purchase_cart_df = df[df['event_type'].isin(['purchase', 'cart'])]

# Contar la cantidad de eventos 'purchase' y 'cart' por hora
purchase_cart_by_hour = purchase_cart_df.groupby(['hour', 'event_type']).size().unstack().fillna(0)

# Visualización
plt.figure(figsize=(12, 6))
purchase_cart_by_hour.plot(kind='bar', stacked=True, color=['orange', 'green'])
plt.title('Cantidad de compras y carritos por hora')
plt.xlabel('Hora')
plt.ylabel('Cantidad de eventos')
plt.legend(title='Tipo de evento')
plt.xticks(rotation=0)
plt.show()


# %%
# Filtrar solo las horas pico (por ejemplo, de 13 a 18)
peak_hours_df = df[df['hour'].isin([13, 14, 15, 16, 17, 18])]

# Top 5 categorías en horas pico
top_categories = peak_hours_df['category_code'].value_counts().nlargest(5)

plt.figure(figsize=(10, 6))
sns.countplot(y='category_code', data=peak_hours_df, order=top_categories.index, palette="viridis")
plt.title('Top 5 categorías en horas pico')
plt.xlabel('Cantidad de eventos')
plt.ylabel('Categoría')
plt.show()

# Top 5 marcas en horas pico
top_brands = peak_hours_df['brand'].value_counts().nlargest(5)

plt.figure(figsize=(10, 6))
sns.countplot(y='brand', data=peak_hours_df, order=top_brands.index, palette="magma")
plt.title('Top 5 marcas en horas pico')
plt.xlabel('Cantidad de eventos')
plt.ylabel('Marca')
plt.show()


# %%
# Contar la cantidad de cada tipo de evento por hora
event_counts_by_hour = df.groupby(['hour', 'event_type']).size().unstack().fillna(0)

# Calcular la proporción de `cart` y `purchase` respecto a `view` para cada hora
event_counts_by_hour['cart_to_view_ratio'] = event_counts_by_hour['cart'] / event_counts_by_hour['view']
event_counts_by_hour['purchase_to_view_ratio'] = event_counts_by_hour['purchase'] / event_counts_by_hour['view']

# Visualización de las proporciones
plt.figure(figsize=(12, 6))
event_counts_by_hour[['cart_to_view_ratio', 'purchase_to_view_ratio']].plot(kind='bar', color=['orange', 'green'])
plt.title('Proporción de carrito y compra respecto a vistas por hora')
plt.xlabel('Hora')
plt.ylabel('Proporción')
plt.legend(title='Tipo de Conversión')
plt.xticks(rotation=0)
plt.show()


# %%





# %%
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# %%
def prepare_data(df):
    """Prepara los datos para el modelo de red neuronal."""
    # Filtrar solo las horas pico
    peak_hours_df = df[df['hour'].isin([13, 14, 15, 16, 17, 18])]
    
    # Crear features temporales
    peak_hours_df['minute'] = peak_hours_df['event_time'].dt.minute
    peak_hours_df['second'] = peak_hours_df['event_time'].dt.second
    
    # Encodificar variables categóricas
    le_category = LabelEncoder()
    le_brand = LabelEncoder()
    le_event = LabelEncoder()
    
    peak_hours_df['category_encoded'] = le_category.fit_transform(peak_hours_df['category_code'])
    peak_hours_df['brand_encoded'] = le_brand.fit_transform(peak_hours_df['brand'])
    peak_hours_df['event_encoded'] = le_event.fit_transform(peak_hours_df['event_type'])
    
    # Crear ventanas temporales de 1 hora
    def create_hourly_aggregations(group):
        return pd.Series({
            'total_events': len(group),
            'view_count': len(group[group['event_type'] == 'view']),
            'cart_count': len(group[group['event_type'] == 'cart']),
            'purchase_count': len(group[group['event_type'] == 'purchase']),
            'unique_categories': group['category_code'].nunique(),
            'unique_brands': group['brand'].nunique(),
            'avg_price': group['price'].mean()
        })
    
    # Agrupar por hora y crear features agregados
    hourly_data = peak_hours_df.groupby(['hour']).apply(create_hourly_aggregations).reset_index()
    
    # Normalizar los datos
    scaler = MinMaxScaler()
    features = ['total_events', 'view_count', 'cart_count', 'purchase_count', 
                'unique_categories', 'unique_brands', 'avg_price']
    hourly_data[features] = scaler.fit_transform(hourly_data[features])
    
    return hourly_data, scaler

# %%
def build_model(input_shape):
    """Construye el modelo de red neuronal."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(5, activation='softmax')  # 5 categorías principales
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# %%
def train_and_evaluate(df):
    """Entrena y evalúa el modelo."""
    # Preparar datos
    hourly_data, scaler = prepare_data(df)
    
    # Preparar features y target
    X = hourly_data[['total_events', 'view_count', 'cart_count', 'purchase_count', 
                     'unique_categories', 'unique_brands', 'avg_price']].values
    
    # Crear target dummy (ejemplo simplificado - ajustar según necesidades)
    y = np.random.randint(0, 5, size=(len(X), 5))  # 5 categorías principales
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construir y entrenar modelo
    model = build_model((X.shape[1],))
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluar modelo
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    return model, history, scaler

# %%
def plot_training_history(history):
    """Visualiza el historial de entrenamiento."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
def make_predictions(model, new_data, scaler):
    """Realiza predicciones con el modelo entrenado."""
    # Preparar nuevos datos
    scaled_data = scaler.transform(new_data)
    
    # Hacer predicción
    predictions = model.predict(scaled_data)
    
    # Obtener la categoría más probable
    predicted_categories = np.argmax(predictions, axis=1)
    
    return predicted_categories, predictions

# %%
#entrenar el modelo 

model, history, scaler = train_and_evaluate(df)



# %%
#visualizar los resultados 
plot_training_history(history)

# %%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

# %%
def evaluate_model_performance(model, X_test, y_test, category_names):
    """
    Evalúa el rendimiento del modelo usando múltiples métricas
    """
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # 1. Métricas básicas
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='weighted')
    
    print("=== Métricas Generales ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # 2. Reporte detallado por categoría
    print("\n=== Reporte Detallado por Categoría ===")
    print(classification_report(y_test_classes, y_pred_classes, target_names=category_names))
    
    # 3. Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=category_names,
                yticklabels=category_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
def evaluate_baseline_comparison(X_test, y_test, y_pred):
    """
    Compara el modelo con un baseline simple (predicción de la clase mayoritaria)
    """
    # Crear baseline (prediciendo siempre la clase más común)
    baseline_pred = np.zeros_like(y_pred)
    majority_class = np.argmax(np.bincount(np.argmax(y_test, axis=1)))
    baseline_pred[:, majority_class] = 1
    
    # Calcular métricas para ambos modelos
    model_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    baseline_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(baseline_pred, axis=1))
    
    print("\n=== Comparación con Baseline ===")
    print(f"Accuracy del modelo: {model_accuracy:.4f}")
    print(f"Accuracy del baseline: {baseline_accuracy:.4f}")
    print(f"Mejora sobre el baseline: {((model_accuracy - baseline_accuracy) / baseline_accuracy * 100):.2f}%")

# %%
def plot_prediction_confidence(y_pred, category_names):
    """
    Visualiza la confianza del modelo en sus predicciones
    """
    confidence_scores = np.max(y_pred, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=50)
    plt.title('Distribución de la Confianza en las Predicciones')
    plt.xlabel('Confianza')
    plt.ylabel('Frecuencia')
    plt.show()
    
    # Confianza promedio por categoría
    predicted_classes = np.argmax(y_pred, axis=1)
    avg_confidence = pd.DataFrame({
        'Categoría': [category_names[i] for i in predicted_classes],
        'Confianza': confidence_scores
    }).groupby('Categoría')['Confianza'].mean()
    
    plt.figure(figsize=(10, 6))
    avg_confidence.plot(kind='bar')
    plt.title('Confianza Promedio por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Confianza Promedio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
def evaluate_time_consistency(model, df, scaler, features):
    """
    Evalúa la consistencia del modelo a lo largo del tiempo
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    for train_idx, test_idx in tscv.split(df):
        # Preparar datos
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        X_train = scaler.transform(train_data[features])
        X_test = scaler.transform(test_data[features])
        
        # Entrenar y evaluar
        model.fit(X_train, train_data['target'])
        accuracy = model.score(X_test, test_data['target'])
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, marker='o')
    plt.title('Consistencia del Modelo a lo Largo del Tiempo')
    plt.xlabel('Split Temporal')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    print("\n=== Consistencia Temporal ===")
    print(f"Accuracy promedio: {np.mean(accuracies):.4f}")
    print(f"Desviación estándar: {np.std(accuracies):.4f}")

# %%
def evaluate_complete_model(model, X_test, y_test, y_pred, category_names, df, scaler, features):
    """
    Ejecuta todas las evaluaciones
    """
    # 1. Evaluación general del modelo
    evaluate_model_performance(model, X_test, y_test, category_names)
    
    # 2. Comparación con baseline
    evaluate_baseline_comparison(X_test, y_test, y_pred)
    
    # 3. Análisis de confianza
    plot_prediction_confidence(y_pred, category_names)
    
    # 4. Evaluación temporal
    evaluate_time_consistency(model, df, scaler, features)

# %%
category_names = ['desconocido', 'electronics.smartphone', 'electronics.clocks', 
                 'electronics.video.tv', 'computers.notebook']

evaluate_complete_model(model, X_test, y_test, y_pred, category_names, df, scaler, features)