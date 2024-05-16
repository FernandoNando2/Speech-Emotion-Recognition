import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np

#load csv
df = pd.read_csv('classification_results.csv')

# create a bar plot
counts = df['Predicted Class'].value_counts()
colors = ['blue', 'green', 'red', 'magenta']

plt.bar(counts.index, counts.values, color=colors[:len(counts)], alpha=0.5)
plt.xlabel('Emotions')
plt.ylabel('Number of Predictions')
plt.title('Speech Emotion Predictions')
plt.yticks(range(0, 26,2))

# add a numerical representation on top of each column
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom', color='black')

plt.show()

fig, axs = plt.subplots(2, 2, figsize=(20, 8))  # Create 2x2 grid of subplots
axs = axs.ravel()  # Flatten the grid to easily index it

classes = ['calm', 'angry', 'fearful', 'sad']  # Define your classes

for i, cls in enumerate(classes):
    file = f"C:/Users/Fernando Hernandez/Desktop/ITQ/8vo Semestre/Ciencia de Datos/Speech Emotions/Test/test-{i+1}.wav"
    y, sr = librosa.load(file)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    librosa.display.specshow(D, sr=sr, ax=axs[i])
    axs[i].set_title(f'Spectrogram for {cls}')

plt.tight_layout()
plt.show()