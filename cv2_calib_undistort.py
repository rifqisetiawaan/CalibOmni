import cv2
import numpy as np

# Load the calibration data
calibration_data = np.load('calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# Load an image
img = cv2.imread('C:/Users/lenovo/Pictures/Camera Roll/WIN_20240629_14_35_51_Pro.jpg')
h, w = img.shape[:2]

# Compute the optimal new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image based on the region of interest
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Save and display the undistorted image
cv2.imwrite('undistorted_image.png', dst)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
