import cv2
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Open the video file (replace 'path_to_test_video.mp4' with your test video file)
cap = cv2.VideoCapture('0001_jpg.rf.890a5f9943244f9472a6e1fd32859ea6.jpg')

# Initialize video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('crowd_detected_output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize lists to store frame numbers and crowd counts
frame_numbers = []
crowd_counts = []

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break

    # Run the YOLO model for person detection (class 0 is 'person')
    result = model.predict(frame, classes=0)
    boxes = result[0].boxes

    count = 0

    for box in boxes:
        count += 1
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        confidence = math.ceil((box.conf[0] * 100)) / 100
        label = f'person: {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    label = f'Crowd Count: {count}'
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Store frame number and crowd count
    frame_numbers.append(frame_number)
    crowd_counts.append(count)
    frame_number += 1

    # Write the frame with bounding boxes and labels to the output video
    out.write(frame)

    # Display the frame with crowd count
    cv2.imshow('Crowd Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# Plot the graph of crowd count over time (frames)
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, crowd_counts, label='Crowd Count')
plt.xlabel('Frames')
plt.ylabel('Count')
plt.title('Crowd Data versus Time')
plt.legend()

# Save the figure as an image
plt.savefig('crowd_count_graph.png')
plt.show()