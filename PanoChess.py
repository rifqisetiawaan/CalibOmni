import cv2
import numpy as np
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (9, 7)  # Number of inner corners per row and column
square_size = 25  # Size of a square in your chosen unit (e.g., 25 mm)

# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real dimensions of the checkerboard squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob(r'pic/panoramic_view.png')  # Use a raw string or forward slashes

# Ensure that there are images found
if not images:
    print("No images found. Check the path and file names.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Check if there are enough points for calibration
if not objpoints or not imgpoints:
    print("Not enough points were found for calibration. Make sure the checkerboard is fully visible in the images.")
    exit()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Optionally, save the calibration result
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
