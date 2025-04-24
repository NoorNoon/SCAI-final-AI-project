import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize detectors
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)


def analyze_facial_landmarks(face_landmarks):
    # Simple emotion detection based on facial landmarks
    # This is a basic implementation - you might want to use a proper emotion detection model
    mouth_points = [landmarks.y for landmarks in face_landmarks.landmark[48:68]]
    eyebrows_points = [landmarks.y for landmarks in face_landmarks.landmark[17:27]]

    mouth_height = max(mouth_points) - min(mouth_points)
    eyebrow_height = max(eyebrows_points) - min(eyebrows_points)

    # Simple threshold-based detection
    if mouth_height > 0.03 and eyebrow_height < 0.02:
        return "Happy"
    return "Not Happy"


def analyze_hand_gesture(hand_landmarks):
    # Simple gesture detection based on thumb up/down
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]

    if thumb_tip.y < thumb_base.y:
        return "Positive"
    return "Negative"


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process face detection
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)

    # Draw face mesh
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            emotion = analyze_facial_landmarks(face_landmarks)
            cv2.putText(image, f"Face: {emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = analyze_hand_gesture(hand_landmarks)
            cv2.putText(image, f"Hand: {gesture}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Emotion Detection', image)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()