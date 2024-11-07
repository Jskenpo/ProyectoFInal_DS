import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class EcommerceAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path and load data."""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        self.model = None
        self.scaler = MinMaxScaler()
    
    def preprocess_data(self):
        """Clean and preprocess the data."""
        # Fill missing values
        self.df['category_code'].fillna('unknown', inplace=True)
        self.df['brand'].fillna('unknown', inplace=True)
        
        # Convert timestamp and extract features
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['hour'] = self.df['event_time'].dt.hour
        self.df['day_of_week'] = self.df['event_time'].dt.dayofweek
        
        # Filter peak hours (13-18)
        self.peak_hours_df = self.df[self.df['hour'].isin(range(13, 19))]
    
    def visualize_data(self):
        """Create data visualizations."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Event distribution by hour
        sns.countplot(data=self.df, x='hour', hue='event_type', ax=axes[0,0])
        axes[0,0].set_title('Event Distribution by Hour')
        
        # Top categories during peak hours
        top_cats = self.peak_hours_df['category_code'].value_counts().nlargest(5)
        sns.barplot(x=top_cats.values, y=top_cats.index, ax=axes[1,0])
        axes[1,0].set_title('Top 5 Categories in Peak Hours')
        
        plt.tight_layout()
        return fig
    
    def build_model(self, input_shape):
        """Build and compile the neural network model."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        return model
    
    def prepare_model_data(self):
        """Prepare data for the neural network model."""
        agg_data = self.peak_hours_df.groupby(['hour', 'day_of_week']).agg({
            'event_type': ['count', 
                           lambda x: (x == 'view').sum(), 
                           lambda x: (x == 'cart').sum(), 
                           lambda x: (x == 'purchase').sum()],
            'category_code': 'nunique',
            'brand': 'nunique',
            'price': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_data.columns = ['hour', 'day_of_week', 'total_events', 'views', 
                            'carts', 'purchases', 'unique_categories', 
                            'unique_brands', 'avg_price', 'price_std']
        
        # Scale features
        feature_cols = ['total_events', 'views', 'carts', 'purchases', 
                        'unique_categories', 'unique_brands', 'avg_price', 'price_std']
        agg_data[feature_cols] = self.scaler.fit_transform(agg_data[feature_cols])
        
        # Create target variable (example: predicting high/low activity periods)
        agg_data['target'] = (agg_data['total_events'] > agg_data['total_events'].quantile(0.75)).astype(int)
        
        return agg_data
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model with early stopping and learning rate reduction."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with multiple metrics."""
        y_pred = self.model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='binary')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title('Confusion Matrix')
        
        axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title('ROC Curve')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'plots': fig
        }
    
    def run_analysis(self):
        data = self.prepare_model_data()
        X = data.drop(['hour', 'day_of_week', 'target'], axis=1)
        y = data['target']
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
        
        self.model = self.build_model((X.shape[1],))
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        evaluation = self.evaluate_model(X_test, y_test)
        return history, evaluation

# Usage example
if __name__ == "__main__":
    analyzer = EcommerceAnalyzer('sampled_data_25k.csv')
    viz_figure = analyzer.visualize_data()
    viz_figure.savefig('data_visualization.png', bbox_inches='tight', dpi=300)
    
    history, evaluation = analyzer.run_analysis()
    
    print("\nModel Performance:")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1']:.4f}")
    print(f"AUC: {evaluation['roc_auc']:.4f}")

    print("\nConfusion Matrix:\n", evaluation['confusion_matrix'])
    
    evaluation['plots'].savefig('model_evaluation.png', bbox_inches='tight', dpi=300)
