import cv2
import mediapipe as mp
import numpy as np

# تحميل نموذج Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # قلب الصورة لتكون مثل المرآة
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    action_detected = "No Fight Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # استخراج مواضع اليد
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand_x, hand_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # حساب السرعة (فرق الحركة بين الإطارات)
            if hand_x in prev_positions:
                speed = np.linalg.norm(np.array(prev_positions[hand_x]) - np.array([hand_x, hand_y]))
                if speed > 30:  # إذا كانت الحركة سريعة جدًا
                    action_detected = "Fight Detected!"

            prev_positions[hand_x] = (hand_x, hand_y)

    # عرض الحالة على الشاشة
    cv2.putText(frame, action_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if action_detected == "Fight Detected!" else (0, 255, 0), 2)

    cv2.imshow("Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

cap.release()
cv2.destroyAllWindows()