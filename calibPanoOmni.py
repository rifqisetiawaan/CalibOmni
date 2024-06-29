#!/usr/bin/env python3

import cv2
import numpy as np
import math
from ultralytics import YOLO

# Muat model YOLOv8
model = YOLO('best.pt')  # Ganti dengan model YOLOv8 yang Anda miliki

def create_panorama(frame):
    height, width = frame.shape[:2]
    
    # Center and radii
    cx, cy = 325, 240
    R_max = min(cx, cy)
    R_min = 80  # Assuming the inner circle radius is zero, adjust as needed

    # Define panoramic dimensions
    X_pano = int(2*((R_max+R_min)/2)*math.pi)
    Y_pano = R_max + R_min

    # Create an empty panoramic image
    pano_img = np.zeros((Y_pano, X_pano, 3), dtype=np.uint8)

    # Iterate over the panoramic image pixels
    for x in range(X_pano):
        for y in range(Y_pano):
            # Calculate polar coordinates
            R = ((y / Y_pano) * (R_max - R_min)) + R_min
            Theta = (x / X_pano) * 2.0 * math.pi

            # Map to circular image coordinates
            X_circ = int(cx + R * math.sin(Theta))
            Y_circ = int(cy + R * math.cos(Theta))

            if 0 <= X_circ < width and 0 <= Y_circ < height:
                pano_img[y, x] = frame[Y_circ, X_circ]

    return pano_img

# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def get_center_coordinates(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return int(x_center), int(y_center)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Create the panoramic image from the current frame
    panorama = create_panorama(frame)
    panorama = cv2.flip(panorama, 0)
    # Deteksi objek dengan YOLOv8
    results = model(panorama)

    # Ambil koordinat titik tengah objek yang terdeteksi
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            x_center, y_center = get_center_coordinates([x_min, y_min, x_max, y_max])
            
            # Gambar kotak dan titik tengah
            cv2.rectangle(panorama, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.circle(panorama, (x_center, y_center), 5, (0, 0, 255), -1)
            cv2.putText(panorama, f'({x_center}, {y_center})', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'({x_center}, {y_center})', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Display the original frame and the panoramic view
    # cv2.imshow('Original Frame', frame)
    cv2.imshow('Panoramic View', panorama)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
