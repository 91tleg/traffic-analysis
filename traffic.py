import cv2
import numpy as np

def process_video(url):
    video = cv2.VideoCapture(url)

    if not video.isOpened():
        print("Error: Cannot open video stream")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Resize
        frame = cv2.resize(frame, (640, 480))

        x, y, w, h = 375, 109, 4, 15
        roi = frame[y:y+h, x:x+w]

        # Convert to HSV for better detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([90, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        status = "No light detected"
        if red_pixels > green_pixels:
            status = "Red light detected"
        elif green_pixels > red_pixels:
            status = "Green light detected"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Traffic Light Detection", frame)
        cv2.imshow("ROI", roi)

        if cv2.waitKey(1) & 0xFF == 27:  # esc
            break
    video.release()
    cv2.destroyAllWindows()

video_url = "https://trafficcams.bellevuewa.gov/traffic-edge/CCTV075L.stream/playlist.m3u8"
process_video(video_url)
