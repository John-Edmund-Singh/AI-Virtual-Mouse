import cv2
import mediapipe as mp  # To detect hand, we use mediapipe
import pyautogui
import time  # Import time for timing clicks

# Initialize video capture and mediapipe hands
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for click tracking
last_click_time = 0
click_threshold = 0.3  # Time threshold for double-click in seconds
click_count = 0

# Sensitivity multiplier for mouse movement
movement_sensitivity = 1.5  # Increase this value to make it faster

# Initialize variables for smoothing
prev_index_x, prev_index_y = screen_width // 2, screen_height // 2  # Start at screen center

# Variables for scroll tracking
prev_finger_y = 0  # Previous y-position for scrolling

# Function to smooth movement
def smooth_move(curr_x, curr_y, prev_x, prev_y, smoothing_factor=0.1):
    return (prev_x + smoothing_factor * (curr_x - prev_x), 
            prev_y + smoothing_factor * (curr_y - prev_y))

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
                    index_x = screen_width / frame_width * x * movement_sensitivity
                    index_y = screen_height / frame_height * y * movement_sensitivity
                    # Ensure the cursor stays within the screen bounds
                    index_x = min(screen_width - 1, max(0, index_x))
                    index_y = min(screen_height - 1, max(0, index_y))
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)

                    # Smooth the mouse movement
                    index_x, index_y = smooth_move(index_x, index_y, prev_index_x, prev_index_y)
                    pyautogui.moveTo(index_x, index_y)

                    # Update previous position for smoothing
                    prev_index_x, prev_index_y = index_x, index_y

                # Track the middle finger (id == 12)
                if id == 12:
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)

            # Check the distance between the index and middle fingers
            if abs(index_y - middle_y) < 50:  # Threshold for click detection
                current_time = time.time()
                
                if current_time - last_click_time <= click_threshold:
                    click_count += 1
                else:
                    click_count = 1  # Reset if time exceeds threshold
                
                last_click_time = current_time
                
                if click_count == 1:
                    print("Single Click...")
                elif click_count == 2:
                    print("Double Clicking...")
                    pyautogui.doubleClick()  # Perform a double-click action
                    click_count = 0  # Reset click count after double click

            # Scroll functionality: track vertical finger movement for scroll up/down
            if abs(index_y - prev_finger_y) > 20:  # Threshold for scroll
                if index_y < prev_finger_y:  # Finger moved up -> Scroll up
                    print("Scrolling Up...")
                    pyautogui.scroll(10)  # Scroll up
                elif index_y > prev_finger_y:  # Finger moved down -> Scroll down
                    print("Scrolling Down...")
                    pyautogui.scroll(-10)  # Scroll down

                # Update previous finger position
                prev_finger_y = index_y

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
