#!/usr/bin/env python3

import cv2
import numpy as np
import math

def create_panorama(frame):
    height, width = frame.shape[:2]
    
    # Center and radii
    cx, cy = 325, 240
    R_max = min(cx, cy)
    R_min = 80  # Adjust as needed

    # Define panoramic dimensions
    X_pano = int(2 * ((R_max + R_min) / 2) * math.pi)
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
    
    # Display the panoramic view
    cv2.imshow('Panoramic View', panorama)

    # Save the panoramic image if 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('panoramic_view.png', panorama)
        print("Panoramic image saved as 'panoramic_view.png'")

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
