import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc.
model.fuse()

# Open your video file
cap = cv2.VideoCapture("D:\\Work\\ieee-hackathon\\sample3.mp4")

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# Perform inference
	results = model(frame)

	# Render results on the frame
	frame = results[0].plot(kpt_line=False)  # This draws bounding boxes on the frame

	# Display the frame
	cv2.imshow('Traffic Detection', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()