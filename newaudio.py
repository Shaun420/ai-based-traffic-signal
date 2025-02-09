import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from pymongo import MongoClient
import pyaudio
import wave
import struct

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
torch.set_num_threads(1)  # Limit to 1 thread

# Google Maps API key
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"  # Replace with your Google Maps API key

yellow_time = 2
priority_time = 60  # Green light duration for the priority lane (when siren detected)

# Initialize YOLOv8 model
model = YOLO('train-5.pt')

# Class indices for vehicles
vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Initialize video capture
cap = cv2.VideoCapture("D:/Work/ieee-hackathon/ambulance.mp4")

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
priority_lane = None

# Function to dynamically adjust signal timings
def adjust_signal_timings(vehicle_count, total_cycle_time=120):
    total_vehicles = sum(vehicle_count.values())
    if total_vehicles == 0:
        return {lane: {"green": total_cycle_time // 4, "red": 0, "yellow": yellow_time} for lane in vehicle_count}

    signal_times = {}
    for lane, count in vehicle_count.items():
        signal_times[lane] = {
            "green": (count / total_vehicles) * total_cycle_time,
            "red": (total_cycle_time - (count / total_vehicles) * total_cycle_time - yellow_time),
            "yellow": yellow_time
        }
    return signal_times

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

# Function to detect siren sound
def detect_siren(stream):
    global siren_detected
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Simple threshold-based detection (customize as needed)
    if np.abs(audio_data).mean() > 500:  # Adjust the threshold as needed
        siren_detected = True
    else:
        siren_detected = False

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Main loop to process video frames and adjust signals
vehicle_count = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0, 'lane_4': 0}

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

    # If a siren is detected, prioritize the emergency vehicle lane
    if siren_detected:
        print("Emergency vehicle detected! Adjusting traffic signals.")
        
        # Detect which lane the ambulance is in based on the bounding box
        for vehicle in vehicles:
            # Assuming an ambulance class is added in vehicle_classes, or use other detection methods
            if int(vehicle.cls) == 5:  # Assuming ambulance class is 5
                center_x = (vehicle.xyxy[0].cpu().numpy()[0] + vehicle.xyxy[0].cpu().numpy()[2]) / 2
                if center_x < frame.shape[1] / 4:
                    priority_lane = 'lane_1'
                elif center_x < frame.shape[1] / 2:
                    priority_lane = 'lane_2'
                elif center_x < 3 * frame.shape[1] / 4:
                    priority_lane = 'lane_3'
                else:
                    priority_lane = 'lane_4'

        # Prioritize the lane where the siren/ambulance is detected
        for lane in signal_times:
            if lane == priority_lane:
                signal_times[lane]["green"] = priority_time  # Set the green light longer for the priority lane
            else:
                signal_times[lane]["green"] = 0  # Stop other lanes temporarily

    # Update traffic signals in the database
    update_traffic_signals(signal_times)

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

# Final plotting of signal times
lanes = list(signal_times.keys())
timings = [x["green"] for x in signal_times.values()]

plt.bar(lanes, timings)
plt.xlabel('Lanes')
plt.ylabel('Green Light Duration (seconds)')
plt.title('Dynamic Traffic Signal Timings Based on Vehicle Count')
plt.show()
