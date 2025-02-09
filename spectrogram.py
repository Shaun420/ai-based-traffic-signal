import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def audio_to_mfcc(audio_path, max_length=100):
    signal, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    # Pad or truncate to ensure consistent shape
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs

def load_audio_files(data_dir):
    X, y = [], []
    labels = os.listdir(data_dir)
    print(f"Found labels: {labels}")  # Debug: print found labels
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            print(f"Processing directory: {label_dir}")  # Debug: print current directory
            for filename in os.listdir(label_dir):
                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    audio_path = os.path.join(label_dir, filename)
                    mfccs = audio_to_mfcc(audio_path)
                    X.append(mfccs)
                    y.append(labels.index(label))  # Label as an integer
    print(f"Total samples loaded: {len(X)}")  # Debug: print number of samples loaded
    return np.array(X), np.array(y)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_and_train_model(X_train, y_train):
    X_train = X_train[..., np.newaxis]  # Adding channel dimension
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')  # Number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    return model

# Specify the input directory
input_directory = 'D:/Work/ieee-hackathon/audio/data/a'  # Replace with your main audio directory

# Process the audio files and split the data
X, y = load_audio_files(input_directory)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train the model
model = build_and_train_model(X_train, y_train)

# Evaluate the model
X_test = X_test[..., np.newaxis]  # Adding channel dimension for test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.2f}')
model.save('D:/Work/ieee-hackathon/models/sirens_model.keras')



