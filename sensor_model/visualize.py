import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Ignore seaborn future warnings for cleaner terminal output
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. Load the Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'rf_model.joblib')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'training_dataset_with_yield.csv')

os.makedirs(FIGURES_DIR, exist_ok=True)

bundle = joblib.load(MODEL_PATH)

classifier = bundle.get('classifier', bundle.get('model'))
regressor = bundle.get('regressor', None)
label_encoder = bundle.get('label_encoder')

FEATURES = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'PH', 'Light_Intensity']

# ==========================================
# GRAPH 1: Feature Importance 
# ==========================================
plt.figure(figsize=(10, 6))
importances = classifier.feature_importances_
sns.barplot(x=importances[:len(FEATURES)], y=FEATURES, hue=FEATURES, palette='viridis', legend=False)
plt.title('Sensor Importance for Disease Classification', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Sensors')
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'Slide_Feature_Importance.png')
plt.savefig(fig1_path)
print(f"Feature Importance graph saved to {fig1_path}!")

# ==========================================
# GRAPH 2 & 3: Using your specific dataset
# ==========================================
try:
    df = pd.read_csv(DATASET_PATH) 
    
    X = df[FEATURES]
    y_class = df['Target_Disease'] 
    y_yield = df.get('Yield_Rate', pd.Series([50.0]*len(df)))
    
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_yield, y_test_yield = train_test_split(X, y_yield, test_size=0.2, random_state=42)

    X_test_scaled = bundle['scaler'].transform(X_test)
    
# --- Graph 2: Confusion Matrix ---
    y_pred_class = classifier.predict(X_test_scaled)
    
    y_test_class_encoded = label_encoder.transform(y_test_class)
    
    cm = confusion_matrix(y_test_class_encoded, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Disease Classification Accuracy (Confusion Matrix)')
    plt.xlabel('Predicted Disease')
    plt.ylabel('Actual Disease')
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, 'Slide_Confusion_Matrix.png')
    plt.savefig(fig2_path)
    print(f"Confusion Matrix saved to {fig2_path}!")

    # --- Graph 3: Actual vs Predicted Yield ---
    if regressor is not None:
        y_pred_yield = regressor.predict(X_test_scaled)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_yield, y_pred_yield, alpha=0.6, color='green')
        
        min_val = min(y_test_yield.min(), y_pred_yield.min())
        max_val = max(y_test_yield.max(), y_pred_yield.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title('Yield Prediction: Actual vs Predicted')
        plt.xlabel('Actual Yield Rate')
        plt.ylabel('Predicted Yield Rate')
        plt.tight_layout()
        fig3_path = os.path.join(FIGURES_DIR, 'Slide_Yield_Regression.png')
        plt.savefig(fig3_path)
        print(f"Yield Regression scatter plot saved to {fig3_path}!")

except FileNotFoundError:
    print(f"Error: Make sure dataset exists at '{DATASET_PATH}'!")