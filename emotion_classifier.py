import joblib
from emotion_recognition import extract_feature
import csv
import os

# Load the saved model
model = joblib.load('7396.pkl')

with open('classification_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["File Name", "Predicted Class"])

    # Iterate over the range of files
    for i in range(1, 61):
        # Load your new audio file
        audio_name = os.path.basename(f"C:/Users/Fernando Hernandez/Desktop/ITQ/8vo Semestre/Ciencia de Datos/Speech Emotions/Test/test-{i}.wav")
        audio = f"C:/Users/Fernando Hernandez/Desktop/ITQ/8vo Semestre/Ciencia de Datos/Speech Emotions/Test/test-{i}.wav"
        features = extract_feature(audio, mfcc=True, chroma=True, mel=True)
        # Predict the class of the audio file
        predicted_class = model.predict([features])
        # Write the file name and the predicted class to the CSV file
        writer.writerow([audio_name, predicted_class[0]])
