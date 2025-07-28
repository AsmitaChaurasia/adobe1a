import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_structure_model(dataset_path: str, model_output_dir: str):
    """
    Trains the classifier on the feature dataset and saves the model.
    """
    try:
        df = pd.read_csv(dataset_path)
        df.dropna(inplace=True)

        # MODIFICATION 1: Remove the 'H4' class before training
        print(f"Original sample count: {len(df)}")
        if 'H4' in df['label'].unique():
            df = df[df['label'] != 'H4']
            print(f"Sample count after removing H4: {len(df)}")

        # MODIFICATION 2: Add the new features to the feature list
        features = [
            'line_size', 'is_bold', 'word_count', 'numbering_pattern',
            'is_centered', 'space_above', 'ratio_of_caps',
            'is_on_first_page', 'vertical_position'
        ]
        X = df[features]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

        print("Training the RandomForestClassifier model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        print("Model training complete.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            
        model_output_path = os.path.join(model_output_dir, 'heading_classifier_gold.joblib')
        joblib.dump(model, model_output_path)
        print(f"\nModel saved successfully to: {model_output_path}")

    except FileNotFoundError:
        print(f"Error: The dataset file was not found at '{dataset_path}'")
        print("Please ensure you have run 'create_dataset.py' first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Corrected path to the dataset
    dataset_path = 'trained_model/gold_standard_dataset.csv'
    model_dir = 'trained_model'
    train_structure_model(dataset_path, model_dir)