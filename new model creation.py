import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from filter_feature import apply_filters, extract_features

def main():
    # --- CONFIGURATION ---
    # Put the exact names of any labels you want to skip in this list. 
    # For example: ['thumb', 'index']
    CLASSES_TO_IGNORE = ['thumb'] 
    
    # 1. Load the Data
    print("Loading master_emg_data.csv...")
    try:
        df = pd.read_csv("master_emg_data.csv")
    except FileNotFoundError:
        print("Error: master_emg_data.csv not found!")
        return

    # --- THE FILTER ---
    # This completely removes those rows from the data before the ML model ever sees it
    if CLASSES_TO_IGNORE:
        print(f"Ignoring the following classes: {CLASSES_TO_IGNORE}")
        df = df[~df['Label'].isin(CLASSES_TO_IGNORE)]

    # 2. Chop into 500ms (100 sample) windows and extract features
    X = []
    y = []

    chunk_size = 100 
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break 
        
        # Only extract if the whole chunk belongs to the same label
        if len(chunk['Label'].unique()) == 1:
            raw_signals = chunk[["Ch1", "Ch2", "Ch3", "Ch4"]].values
            label = chunk['Label'].iloc[0]
            
            # Filter and Extract using your math engine
            filtered_signals = apply_filters(raw_signals)
            features = extract_features(filtered_signals)
            
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # 3. Train / Test Split with STRATIFICATION
    # test_size=0.15 preserves 85% of your data for teaching the ML model
    # stratify=y guarantees proportional representation of all classes in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    print(f"\nDataset Ready: Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Print out exactly how many windows of each class the ML model is learning from
    print("\nClass distribution in Training Set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f" - {u.upper()}: {c} windows")

    # 4. Train the Model
    print("\nTraining KNN Model...")
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- MODEL ACCURACY ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # 6. Save the Model
    joblib.dump(model, 'emg_knn_model.pkl')
    print("\nModel saved to 'emg_knn_model.pkl'. You are ready to run predict.py!")

if __name__ == "__main__":
    main()