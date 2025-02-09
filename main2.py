import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from pymongo import MongoClient
torch.set_num_threads(1)  # Limit to 1 thread

yellow_time = 2

# Initialize YOLOv8 model
model = YOLO('train-5.pt')

# Class indices for vehicles (COCO dataset: car, motorcycle, bus, truck)
vehicle_classes = [2, 3, 5, 7]

# Initialize video capture
cap = cv2.VideoCapture("D:\\Work\\ieee-hackathon\\sample3.mp4")

# Connect to MongoDB
db_client = MongoClient("mongodb://localhost:27017/")

# Access traffic database
db = db_client["traffic"]

# Access traffic collection
traffic_col = db["traffic"]

# Function to dynamically adjust signal timings
def adjust_signal_timings(vehicle_count, total_cycle_time=120):
	total_vehicles = sum(vehicle_count.values())
	
	# Avoid division by zero
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

# Main loop to process video frames and adjust signals
vehicle_count = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0, 'lane_4': 0}

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# Run YOLOv8 on the frame
	results = model(frame)
	detected_objects = results[0].boxes
	vehicles = [obj for obj in detected_objects if int(obj.cls) in vehicle_classes]

	# Reset vehicle counts for each frame
	vehicle_count = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0, 'lane_4': 0}

	# Classify detected vehicles into different lanes
	for vehicle in vehicles:
		bbox = vehicle.xyxy[0].cpu().numpy()  # Get bounding box
		center_x = (bbox[0] + bbox[2]) / 2    # Calculate center x-coordinate
		
		if center_x < frame.shape[1] / 4:
			vehicle_count['lane_1'] += 1
		elif center_x < frame.shape[1] / 2:
			vehicle_count['lane_2'] += 1
		elif center_x < 3 * frame.shape[1] / 4:
			vehicle_count['lane_3'] += 1
		else:
			vehicle_count['lane_4'] += 1

	# Calculate signal timings
	signal_times = adjust_signal_timings(vehicle_count)
	print(signal_times["lane_1"]["green"])
	print(traffic_col.Maharashtra.Pune.Shivajinagar.RTO[0]["green"])
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[0].green = signal_times["lane_1"]["green"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[0].red = signal_times["lane_1"]["red"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[0].yellow = signal_times["lane_1"]["yellow"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[1].green = signal_times["lane_2"]["green"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[1].red = signal_times["lane_2"]["red"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[1].yellow = signal_times["lane_2"]["yellow"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[2].green = signal_times["lane_3"]["green"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[2].red = signal_times["lane_3"]["red"]
	traffic_col.Maharashtra.Pune.Shivajinagar.RTO[2].yellow = signal_times["lane_3"]["yellow"]
	# Display signal timings on the video frame
	for i, (lane, time) in enumerate(signal_times.items()):
		cv2.putText(frame, f'{lane}: {int(time["green"])}s', (50, 50 + 50 * i), 
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	# Plot bounding boxes for the detected vehicles
	annotated_frame = results[0].plot()

	# Show the frame with detections and signal times
	cv2.imshow('Traffic Detection with Dynamic Signal Timing', annotated_frame)

	# Exit on 'q' key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

db_client.close()

# Plotting the final signal times after the video processing
lanes = list(signal_times.keys())
timings = [x["green"] for x in signal_times.values()]

plt.bar(lanes, timings)
plt.xlabel('Lanes')
plt.ylabel('Green Light Duration (seconds)')
plt.title('Dynamic Traffic Signal Timings Based on Vehicle Count')
plt.show()
