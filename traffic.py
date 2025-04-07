import cv2
import numpy as np
import time
import threading

camera_configs = {
    "164th_24th": {
        "url":  "https://trafficcams.bellevuewa.gov/traffic-edge/CCTV075L.stream/playlist.m3u8",
        "roi": (375, 111, 6, 15),
        "map_pos": (100, 100)
    },
    "164th_northup": {
        "url": "https://trafficcams.bellevuewa.gov/traffic-edge/CCTV076L.stream/playlist.m3u8",
        "roi": (330, 45, 6, 18),
        "map_pos": (100, 100)
    }
}

light_statuses = {name: "None" for name in camera_configs}

def detect_light(url, roi, name):
    video = cv2.VideoCapture(url)
    if not video.isOpened():
        print("Error: Cannot open video stream")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.resize(frame, (640, 480))
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV) # Convert to HSV for better detection

        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([179, 255, 255])
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([90, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        if red_pixels > green_pixels:
            status = "Red"
        elif green_pixels > red_pixels:
            status = "Green"
        else:
            status = "None"
        time.sleep(0.1)



for name, config in camera_configs.items():
    threading.Thread(target=detect_light, args=(config["url"], config["roi"], name), daemon=True).start()

map_img = cv2.imread("bellevue_map.png")
while True:
    display_map = map_img.copy()

    for name, config in camera_configs.items():
        pos = config["map_pos"]
        status = light_statuses[name]
        if status == "Red":
            color = (0, 0, 255)
        elif status == "Green":
            color = (0, 255, 0)
        else: 
            color = (200, 200, 200) # Gray

        cv2.circle(display_map, pos, 15, color, -1)
        cv2.putText(display_map, name, (pos[0] - 30, pos[1] - 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Traffic Light Map", display_map)
    if cv2.waitKey(100) & 0xFF == 27:
        break

cv2.destroyAllWindows()