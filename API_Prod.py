import pandas as pd
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo SVM y el scaler guardados previamente
svm_model = joblib.load('svm_model.pkl')  # Cargar el modelo SVM
scaler = joblib.load('scaler.pkl')  # Cargar el scaler

# Leer las columnas desde el archivo 'columns.txt'
with open('columns.txt', 'r') as file:
    columns = [line.strip() for line in file.readlines()]

# Crear la app de Flask
app = Flask(__name__)

# Función para crear el vector de características con OneHotEncoding
def create_feature_vector(price, brand, category):
    # Crear un diccionario con las 1369 columnas, inicialmente todas como False
    feature_vector = {column: False for column in columns}
    
    # Asignar el valor del precio (price) a la columna correspondiente
    feature_vector['price'] = price
    
    # Asignar True o False a las columnas correspondientes de brand
    brand_column = f'brand_{brand.lower()}'
    if brand_column in feature_vector:
        feature_vector[brand_column] = True
    
    # Asignar True o False a las columnas correspondientes de category_code
    category_column = f'category_code_{category.lower()}'
    if category_column in feature_vector:
        feature_vector[category_column] = True
    
    return feature_vector

# Rutas y funciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados como JSON
    data = request.get_json()

    # Asegurarse de que 'data' es una lista de diccionarios
    if isinstance(data, dict):
        data = [data]  # Envolver el diccionario en una lista para Pandas

    # Crear el DataFrame de las características
    feature_vectors = []
    for item in data:
        price = item.get('price')
        brand = item.get('brand')
        category = item.get('category')
        
        # Crear el vector de características para cada entrada
        feature_vector = create_feature_vector(price, brand, category)
        feature_vectors.append(feature_vector)

    # Convertir los diccionarios a un DataFrame
    df = pd.DataFrame(feature_vectors)

    # Normalizar la columna 'price' usando el scaler cargado
    df['price'] = scaler.transform(df[['price']])

    # Realizar predicciones con el modelo SVM
    y_pred = svm_model.predict(df)

    # Enviar la respuesta en formato JSON
    return jsonify({'predictions': y_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
