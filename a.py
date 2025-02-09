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

# Class indices for vehicles
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
def apply_bandpass_filter(data, target_freq):
    lowcut = target_freq * 0.8
    highcut = target_freq * 1.2
    b, a = butter_bandpass(lowcut, highcut, RATE)
    return lfilter(b, a, data)

# Function to normalize audio
def normalize_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    return audio_data / max_val if max_val > 0 else audio_data

# Function to extract MFCCs for audio classification
def audio_to_mfcc(audio_data):
    if len(audio_data) == 0:
        print("Warning: Received empty audio data. Returning zeros.")
        return np.zeros((13, 100, 1))  # Return zeros if input is empty

    audio_data = audio_data.astype(np.float32)

    n_fft = 512 if len(audio_data) < 2048 else 2048
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13, n_fft=n_fft)

    # Ensure MFCCs have at least 100 frames
    if mfccs.shape[1] < 100:
        mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), 'constant')
    else:
        mfccs = mfccs[:, :100]  # Trim to 100 frames if necessary

    # Calculate delta features only if there are enough frames
    if mfccs.shape[1] > 1:
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_delta = librosa.feature.delta(mfccs, order=2)

        # Combine MFCCs, deltas, and delta-deltas
        combined_mfccs = np.concatenate((mfccs, mfccs_delta, mfccs_delta_delta), axis=0)
    else:
        combined_mfccs = mfccs  # If not enough frames, skip delta features

    # Ensure combined_mfccs has the right number of elements (13, 100, 1)
    if combined_mfccs.shape[0] < 13:
        combined_mfccs = np.pad(combined_mfccs, ((0, 13 - combined_mfccs.shape[0]), (0, 0)), 'constant')
    elif combined_mfccs.shape[0] > 13:
        combined_mfccs = combined_mfccs[:13, :]  # Trim if more than 13

    combined_mfccs = combined_mfccs.reshape((13, 100, 1))  # Reshape to expected dimensions

    return combined_mfccs  # Return with the correct shape

# Function to detect siren
def detect_siren(stream):
    global siren_detected, detected_siren_type
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Apply bandpass filter for ambulance frequencies
    filtered_audio = apply_bandpass_filter(audio_data, target_freq=1000)  # Target frequency around 1000 Hz

    # Normalize audio
    filtered_audio = normalize_audio(filtered_audio)

    # Use the trained model to predict siren sound
    mfccs = audio_to_mfcc(filtered_audio)
    mfccs = mfccs.reshape(1, 13, 100, 1)
    prediction = sound_model.predict(mfccs)

    # Assuming the model outputs probabilities for classes: ['no_siren', 'ambulance', 'firetruck', 'police', 'honk']
    siren_classes = ['no siren', 'ambulance', 'firetruck', 'police', 'honk']
    detected_class = np.argmax(prediction)

    # Check if ambulance or any siren is detected
    if detected_class in [1, 2, 3]:  # If ambulance, firetruck, or police detected
        siren_detected = True
        detected_siren_type = siren_classes[detected_class]
        print(f"{detected_siren_type} detected! Adjusting traffic signals.")
    elif detected_class == 4:  # If honk detected
        siren_detected = False
        detected_siren_type = None
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
            
            affected_lane = f'lane_{int(center_x // (frame.shape[1] / 4) + 1)}'
            signal_times[affected_lane]['green'] = 15  # Give priority green time
            signal_times[affected_lane]['red'] = 0

            # Set the yellow time for the affected lane
            for lane in signal_times.keys():
                if lane != affected_lane:
                    signal_times[lane]['yellow'] = yellow_time  # Keep yellow time for other lanes

    # Update traffic signals in the database
    update_traffic_signals(signal_times)

    # Store normal signal times for later plotting
    normal_signal_times.append({lane: signal_times[lane] for lane in signal_times})

    # Display the frame
    cv2.imshow("Traffic Monitoring", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()  # Close all OpenCV windows

# Plotting traffic signal timings
time_intervals = np.arange(len(normal_signal_times))
green_times = [signal[lane]['green'] for signal in normal_signal_times for lane in signal]
red_times = [signal[lane]['red'] for signal in normal_signal_times for lane in signal]
yellow_times = [signal[lane]['yellow'] for signal in normal_signal_times for lane in signal]

# Calculate average green light durations for each lane
lanes = list(signal_times.keys())
normal_timings = []
if normal_signal_times:
    for lane in lanes:
        lane_green_times = [sig[lane]['green'] for sig in normal_signal_times if lane in sig]
        normal_timings.append(np.mean(lane_green_times) if lane_green_times else 0)
else:
    normal_timings = [0] * len(lanes)

# Initialize emergency timings
emergency_timings = [0] * len(lanes)
if siren_detected and vehicles:
    last_vehicle = vehicles[-1]
    bbox = last_vehicle.xyxy[0].cpu().numpy()
    center_x = (bbox[0] + bbox[2]) / 2
    affected_lane_index = int(center_x // (frame.shape[1] / 4))

    if affected_lane_index < len(emergency_timings):
        emergency_timings[affected_lane_index] = max(signal_times[f'lane_{affected_lane_index + 1}']['green'] + 5, 20)  # 20 seconds minimum for emergency

# Now plotting
colors = ['green' if lane == 'lane_4' else 'red' for lane in lanes]
plt.figure(figsize=(12, 6))
plt.bar(lanes, normal_timings, label='Normal Timing', alpha=0.6, color=colors)
plt.bar(lanes, emergency_timings, label='Emergency Timing', alpha=0.6, color='green')

plt.xlabel('Lanes')
plt.ylabel('Green Light Duration (seconds)')
plt.title('Dynamic Traffic Signal Timings Based on Vehicle Count')
plt.legend()
plt.show()
