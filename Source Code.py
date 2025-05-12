import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr
import os
from PIL import Image

# Load the dataset
try:
    df = pd.read_csv('english.csv')
    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df['label'].value_counts())
except FileNotFoundError:
    print("Error: 'english.csv' not found. Please check the file path.")
    # Instead of exiting, assign an empty DataFrame to df
    df = pd.DataFrame()  # This allows the code to continue without crashing

# Preprocessing: Encode labels
# Check if df is empty before proceeding
if not df.empty:
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    # Simulate image features (replace with actual image processing in real implementation)
    np.random.seed(42)
    num_samples = len(df)
    num_features = 100  # Simulating 100 image features
    X = np.random.rand(num_samples, num_features)
    y = df['label_encoded']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print("Skipping model training and evaluation due to missing dataset.")


# Prediction function for Gradio
def predict_character(image):
    """
    Placeholder function for character prediction.
    In a real implementation, this would:
    1. Preprocess the image (resize, normalize, etc.)
    2. Extract features
    3. Make prediction using the trained model
    """
    try:
        # For demonstration, return a random prediction
        # Use a default list of characters if label_encoder is not available
        characters = label_encoder.classes_ if 'label_encoder' in locals() else list("abcdefghijklmnopqrstuvwxyz0123456789")
        random_pred = np.random.choice(characters)
        return {"Prediction": random_pred, "Confidence": np.random.rand()}
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Prepare example images (if available)
example_images = []
if os.path.exists('example_images'):
    example_images = [["example_images/" + f] for f in os.listdir('example_images')
                    if f.endswith(('.png', '.jpg', '.jpeg'))][:3]

# Create Gradio interface
interface = gr.Interface(
    fn=predict_character,
    inputs=gr.Image(label="Upload Character Image", type="pil"),
    outputs=gr.Label(label="Prediction Results"),
    title="English Character Recognition System",
    description="Upload an image of an English letter (A-Z, a-z) or digit (0-9) for classification",
    examples=example_images if example_images else None,
    allow_flagging="never"
)

# Launch the interface
print("\nLaunching the Gradio interface...")
interface.launch()
