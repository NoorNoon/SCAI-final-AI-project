import cv2
import mediapipe as mp
import time
import os
import math
import winsound

# إعدادات MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
# أطراف الأصابع
finger_tips = [8, 12, 16, 20]
thumb_tip = 4
# تحميل صورة القلب
heart_img = cv2.imread("imagesForYolo/heart.jpeg", cv2.IMREAD_UNCHANGED)  # صورة بخلفية شفافة
# دالة لقياس المسافة بين نقطتين
def distance(pt1, pt2):
   return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
# دالة لحساب حالة الأصابع
def get_finger_status(hand_landmarks, hand_label, img):
   h, w, _ = img.shape
   lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
   fingers = []
   if hand_label == "Right":
       fingers.append(lm_list[thumb_tip][0] > lm_list[thumb_tip - 1][0])
   else:
       fingers.append(lm_list[thumb_tip][0] < lm_list[thumb_tip - 1][0])
   for tip in finger_tips:
       fingers.append(lm_list[tip][1] < lm_list[tip - 2][1])
   return fingers, lm_list
# حفظ صورة
def save_image(img):
   filename = f"Captured/five_fingers_{int(time.time())}.png"
   cv2.imwrite(filename, img)
# لصق صورة القلب على الشاشة
def overlay_image(bg, overlay, x, y):
   h, w = overlay.shape[:2]
   if x + w > bg.shape[1] or y + h > bg.shape[0]:
       return bg
   for i in range(h):
       for j in range(w):
           if overlay[i, j][3] != 0:
               bg[y + i, x + j] = overlay[i, j][:3]
   return bg
# تشغيل الكاميرا
cap = cv2.VideoCapture(0)
if not os.path.exists("Captured"):
   os.makedirs("Captured")
while True:
   success, img = cap.read()
   if not success:
       break
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   result = hands.process(img_rgb)
   hand_data = []
   if result.multi_hand_landmarks and result.multi_handedness:
       for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
           label = hand_label.classification[0].label
           fingers_status, lm_list = get_finger_status(hand_landmarks, label, img)
           finger_count = fingers_status.count(True)
           mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           cv2.putText(img, f'{label} Hand: {finger_count}', (10, 60 if label == "Right" else 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
           # حفظ البيانات لمعالجة اليدين
           hand_data.append({"label": label, "lm": lm_list, "fingers": fingers_status})
           # حالة: علامة السلام
           if fingers_status == [False, True, True, False, False]:
               cv2.putText(img, 'hellooooo!!!!', (200, 400), cv2.FONT_HERSHEY_SIMPLEX,
                           2.2, (255, 0, 255), 5)
           # حالة: رفع 5 أصابع
           if finger_count == 5:
               try:
                   winsound.Beep(1000, 300)
               except:
                   print("تنبيه: لم يتم تشغيل الصوت")
               save_image(img)
           # حالة: قلب بالإصبع (إبهام وسبابة قريبين)
           dist = distance(lm_list[4], lm_list[8])
           if dist < 40:
               img = overlay_image(img, heart_img, lm_list[4][0], lm_list[4][1])
       # حالة: قلب باليدين (قرب إبهامي اليدين)
       if len(hand_data) == 2:
           thumb1 = hand_data[0]["lm"][4]
           thumb2 = hand_data[1]["lm"][4]
           dist_between_thumbs = distance(thumb1, thumb2)
           if dist_between_thumbs < 80:
               # نعرض صورة قلب كبيرة في النص
               center_x = int(img.shape[1] / 2 - heart_img.shape[1] / 2)
               center_y = int(img.shape[0] / 2 - heart_img.shape[0] / 2)
               img = overlay_image(img, heart_img, center_x, center_y)
   cv2.imshow("AI Hand Detector", img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()