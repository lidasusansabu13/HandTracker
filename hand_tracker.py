import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=5)
previous_time = 0
current_time = 0 

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    if results.multi_hand_landmarks:
        for hand_land_marks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_land_marks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                if id == 4:
                    cv2.circle(img,(cx,cy), 25, (255,0, 255), cv2.FILLED) # highlight one point
            mp_drawing.draw_landmarks(img, hand_land_marks, mp_hands.HAND_CONNECTIONS)
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.imshow('Image', img)
    cv2.waitKey(1)
