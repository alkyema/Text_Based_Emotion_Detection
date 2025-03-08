import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder

# 1. Load the Trained Model
model = load_model("lstm_emotion_model.h5")

# 2. Load the Tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_data = f.read()  # Read the file as a string

tokenizer = tokenizer_from_json(tokenizer_data)

# 3. Load the Label Encoder
# If you saved the LabelEncoder classes during training, load them
# Load the LabelEncoder classes
label_classes = np.load("label_classes.npy", allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes


# 4. Define the Prediction Function
def predict_emotion(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")  # Use the same max_length as during training
    
    # Predict the emotion
    prediction = model.predict(padded_sequence)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]

# 5. Test the Model
while True:
    user_input = input("Enter a sentence to predict the emotion (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    predicted_emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {predicted_emotion}")
