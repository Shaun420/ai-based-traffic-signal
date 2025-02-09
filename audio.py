import os
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pyaudio
import tensorflow as tf
import librosa
from scipy.signal import butter, lfilter

# Load the trained model
sound_model = tf.keras.models.load_model('D:/Work/ieee-hackathon/models/sirens_model.keras')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
torch.set_num_threads(1)  # Limit to 1 thread
yellow_time = 2

# Initialize YOLOv8 model
model = YOLO('train-5.pt')

# Class indices for vehiclesq
vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Initialize video capture
cap = cv2.VideoCapture("D:/Work/ieee-hackathon/audio/data/honk.mp4")

# Connect to MongoDB
db_client = MongoClient("mongodb://localhost:27017/")
db = db_client["traffic"]
traffic_col = db["traffic"]

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
siren_detected = False
detected_siren_type = None  # Variable to store detected siren type

# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Updated bandpass filter for ambulance sirens
def apply_bandpass_filter(data, lowcut=500.0, highcut=2000.0):
    b, a = butter_bandpass(lowcut, highcut, RATE)
    return lfilter(b, a, data)

# Function to extract MFCCs for audio classification
def audio_to_mfcc(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
    return np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), 'constant') if mfccs.shape[1] < 100 else mfccs

# Function to detect ambulance siren sound using the loaded model
def detect_siren(stream):
    global siren_detected, detected_siren_type
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Apply bandpass filter for ambulance frequencies
    filtered_audio = apply_bandpass_filter(audio_data)

    # Convert to float and normalize
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 0:
        filtered_audio = filtered_audio.astype(np.float32) / max_val

    # Use the trained model to predict siren sound
    mfccs = audio_to_mfcc(filtered_audio)
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)  # Reshape for model input
    prediction = sound_model.predict(mfccs)

    # Assuming the model outputs probabilities for classes: ['no_siren', 'ambulance', 'firetruck', 'police']
    siren_classes = ['no siren', 'ambulance', 'firetruck', 'police']
    detected_class = np.argmax(prediction)

    # Check if ambulance is detected
    if detected_class == 1:  # If the detected class is 'ambulance'
        siren_detected = True
        detected_siren_type = siren_classes[detected_class]
        print(f"Ambulance detected! Adjusting traffic signals.")
    else:
        siren_detected = False
        detected_siren_type = None

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Function to update traffic signals in MongoDB
def update_traffic_signals(signal_times):
    try:
        for i, lane in enumerate(signal_times.keys()):
            traffic_col.update_one(
                {"location": "Maharashtra.Pune.Shivajinagar.RTO"},
                {"$set": {
                    f"RTO.{i}.green": signal_times[lane]["green"],
                    f"RTO.{i}.red": signal_times[lane]["red"],
                    f"RTO.{i}.yellow": signal_times[lane]["yellow"],
                }}
            )
    except Exception as e:
        print(f"Error updating database: {e}")

# Function to dynamically adjust signal timings
def adjust_signal_timings(vehicle_count, total_cycle_time=120):
    total_vehicles = sum(vehicle_count.values())
    signal_times = {}
    if total_vehicles == 0:
        for lane in vehicle_count.keys():
            signal_times[lane] = {"green": total_cycle_time // len(vehicle_count), "red": 0, "yellow": yellow_time}
        return signal_times

    for lane, count in vehicle_count.items():
        green_time = max(0, (count / total_vehicles) * total_cycle_time)
        signal_times[lane] = {
            "green": green_time,
            "red": max(0, total_cycle_time - green_time - yellow_time),
            "yellow": yellow_time
        }
    return signal_times

# Main loop to process video frames and adjust signals
vehicle_count = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0, 'lane_4': 0}
normal_signal_times = []
emergency_signal_times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect siren sound
    detect_siren(stream)

    # Run YOLOv8 on the frame
    results = model(frame)
    detected_objects = results[0].boxes
    vehicles = [obj for obj in detected_objects if int(obj.cls) in vehicle_classes]

    vehicle_count = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0, 'lane_4': 0}

    for vehicle in vehicles:
        bbox = vehicle.xyxy[0].cpu().numpy()
        center_x = (bbox[0] + bbox[2]) / 2

        if center_x < frame.shape[1] / 4:
            vehicle_count['lane_1'] += 1
        elif center_x < frame.shape[1] / 2:
            vehicle_count['lane_2'] += 1
        elif center_x < 3 * frame.shape[1] / 4:
            vehicle_count['lane_3'] += 1
        else:
            vehicle_count['lane_4'] += 1

    # Adjust traffic signals based on vehicle counts
    signal_times = adjust_signal_timings(vehicle_count)

    # If a siren is detected, prioritize the emergency vehicle
    if siren_detected:
        print(f"Detected siren type: {detected_siren_type}")

        # Set the bypass for the lane with the emergency vehicle
        if vehicles:
            last_vehicle = vehicles[-1]
            bbox = last_vehicle.xyxy[0].cpu().numpy()
            center_x = (bbox[0] + bbox[2]) / 2
            
            # Identify the lane of the emergency vehicle
            affected_lane = f'lane_{int(center_x // (frame.shape[1] / 4) + 1)}'
            signal_times[affected_lane]['green'] = 15
            signal_times[affected_lane]['red'] = 0

            # Set the yellow time for the affected lane
            for lane in signal_times.keys():
                if lane != affected_lane:
                    signal_times[lane]['yellow'] = yellow_time  # Keep yellow time for other lanes

    # Update traffic signals in the database
    update_traffic_signals(signal_times)

    # Store the normal signal times for plotting later
    normal_signal_times.append(signal_times.copy())  # Store the state before possible adjustment

    # Display signal timings on the video frame
    for i, (lane, time) in enumerate(signal_times.items()):
        cv2.putText(frame, f'{lane}: {int(time["green"])}s', (50, 50 + 50 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Plot bounding boxes for the detected vehicles
    annotated_frame = results[0].plot()

    # Show the frame with detections and signal times
    cv2.imshow('Traffic Detection with Dynamic Signal Timing', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
db_client.close()

# After processing frames and adjusting signals
# Final plotting of signal times
lanes = list(signal_times.keys())
normal_timings = [np.mean([sig[lane]['green'] for sig in normal_signal_times]) for lane in lanes]

# Initialize emergency timings
emergency_timings = [0] * len(lanes)

# If a siren was detected, set the emergency timings for the affected lane
if siren_detected and vehicles:
    last_vehicle = vehicles[-1]
    bbox = last_vehicle.xyxy[0].cpu().numpy()
    center_x = (bbox[0] + bbox[2]) / 2
    affected_lane_index = int(center_x // (frame.shape[1] / 4))

    if affected_lane_index < len(emergency_timings):
        emergency_timings[affected_lane_index] = signal_times[f'lane_{affected_lane_index + 1}']['green']

# Plot the normal signal timings
plt.figure(figsize=(12, 6))
colors = ['green' if lane == 'lane_4' else 'red' for lane in lanes]
plt.bar(lanes, normal_timings, label='Normal Timing', alpha=0.6, color=colors)

# Plot the emergency timings
plt.bar(lanes, emergency_timings, label='Emergency Timing', alpha=0.6, color='green')

plt.xlabel('Lanes')
plt.ylabel('Green Light Duration (seconds)')
plt.title('Dynamic Traffic Signal Timings Based on Vehicle Count')
plt.legend()
plt.show()









