import cv2
import mediapipe as mp  # To detect hand, we use mediapipe
import pyautogui

# Initialize video capture and mediapipe hands
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    frame_height, frame_width, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    # Variables to store the index and middle positions
    index_x, index_y = 0, 0
    middle_x, middle_y = 0, 0

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Track the index finger (id == 8)
                if id == 8:
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    pyautogui.moveTo(index_x, index_y)

                # Track the middle finger (id == 12)
                if id == 12:
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)

            # Check the distance between the index and middle fingers
            if abs(index_y - middle_y) < 50:  # Increased threshold
                print("Clicking...")
                pyautogui.click()  # Perform a click action

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
