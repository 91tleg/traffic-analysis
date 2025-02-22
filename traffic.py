import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

image_url = "https://products.issaquahwa.gov/cams/Gilman12th_NW.jpg"


#https://products.issaquahwa.gov/cams/Gilman12th_NW.jpg
#https://products.issaquahwa.gov/cams/FrontGilman_NW.jpg

response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image = image.convert("RGB")

image_np = np.array(image)
image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

x1, y1, x2, y2 = 748, 420, 760, 440
roi = image_np[y1:y2, x1:x2]

# Convert ROI to HSV for color detection
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255)], 
    "Yellow": [(20, 100, 100), (40, 255, 255)], 
    "Green": [(40, 50, 50), (80, 255, 255)]
}

red_pixel_count = 0
green_pixel_count = 0
yellow_pixel_count = 0

for color, (lower, upper) in color_ranges.items():
    lower = np.array(lower)
    upper = np.array(upper)

    # Mask to detect the color in the ROI
    mask = cv2.inRange(roi_hsv, lower, upper)
    
    pixel_count = np.sum(mask > 0)

    if color == "Red":
        red_pixel_count = pixel_count
    elif color == "Green":
        green_pixel_count = pixel_count
    elif color == "Yellow":
        yellow_pixel_count = pixel_count

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_np, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (255, 255, 255), 2)
            cv2.putText(image_np, color, (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"Detected {color} light at [{x + x1}, {y + y1}, {w}, {h}]")

if red_pixel_count > green_pixel_count and red_pixel_count > yellow_pixel_count:
    result = "Red"
elif green_pixel_count > red_pixel_count and green_pixel_count > yellow_pixel_count:
    result = "Green"
elif yellow_pixel_count > red_pixel_count and yellow_pixel_count > green_pixel_count:
    result = "Yellow"
else:
    result = "No light detected"

print(f"Traffic light color: {result}")