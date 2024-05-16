import joblib
import soundfile
import librosa
import numpy as np
import csv
import os

# Load the saved model
model = joblib.load('7396.pkl')

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            mfccs = mfccs.flatten()
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            chroma=chroma.flatten()
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            mel=mel.flatten()
            result=np.hstack((result, mel))
    return result[:180]

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
