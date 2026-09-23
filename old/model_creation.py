import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("Loading dataset...")
    try:
        # 1. Load the data
        df = pd.read_csv("training_features.csv")
    except FileNotFoundError:
        print("Error: training_features.csv not found. Run the extraction script first!")
        return

    # 2. Separate Features (X) from Labels (y)
    # X will be our 8 columns of numbers (Avg and Max for Ch1-4)
    X = df.drop("Label", axis=1)
    # y will be our text labels (e.g., 'index', 'rest', 'thumb')
    y = df["Label"]

    # 3. Split into Training and Testing Sets
    # We use 80% of the data to teach the ML model, and hide 20% to test it later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale the Data (CRITICAL FOR KNN)
    # KNN measures the physical distance between data points. If one feature is measured 
    # in hundreds and another in decimals, the math breaks. Scaling evens the playing field.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train the k-Nearest Neighbors Model
    print("Training the KNN model...")
    # 'n_neighbors=5' means the ML model looks at the 5 closest known data points to guess a new one
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 6. Test the Model
    print("\n--- MODEL EVALUATION ---")
    predictions = knn.predict(X_test_scaled)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    # Print a detailed breakdown of how well it recognized each individual finger
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 7. Save the Model and Scaler for Real-Time Use
    # We save these as .pkl files so our final live-streaming script can load them
    joblib.dump(knn, "emg_knn_model.pkl")
    joblib.dump(scaler, "emg_scaler.pkl")
    print("\nSuccess! Model saved as 'emg_knn_model.pkl'")
    print("Scaler saved as 'emg_scaler.pkl'")

if __name__ == "__main__":
    main()