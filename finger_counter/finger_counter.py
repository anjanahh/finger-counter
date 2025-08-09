import cv2
import numpy as np
import math

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert ROI to HSV for skin color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Draw the contour
        cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

        # Convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

        # Convexity defects
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    # Calculate distance using cosine rule
                    a = math.dist(start, end)
                    b = math.dist(start, far)
                    c = math.dist(end, far)
                    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57

                    # If angle < 90, count as a finger
                    if angle <= 90:
                        finger_count += 1
                        cv2.circle(roi, far, 4, [0, 0, 255], -1)

                # Fingers = defects + 1
                cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frames
    cv2.imshow('Mask', mask)
    cv2.imshow('Finger Counter', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
