import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (9, 7)  # Number of inner corners per row and column
square_size = 25  # Size of a square in your chosen unit (e.g., 25 mm)

# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real dimensions of the checkerboard squares
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load images using a raw string to avoid unicode escape errors
images = glob.glob(r'C:\Users\lenovo\Pictures\Camera Roll\WIN_20240629_14_35_51_Pro.jpg')  # Adjust the path and file extension as needed
# images = glob.glob(r'D:/TA/CalibOmni3/panoramic_view.png')

if not images:
    print("No images found. Check the path and file names.")
    exit()

print(f"Found {len(images)} images.")

for fname in images:
    print(f"Processing image: {fname}")
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"Chessboard corners found in image: {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(10000)
    else:
        print(f"Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()

# Check if there are enough points for calibration
if not objpoints or not imgpoints:
    print("Not enough points were found for calibration. Make sure the checkerboard is fully visible in the images.")
    exit()

print(f"Collected {len(objpoints)} object points and {len(imgpoints)} image points.")

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Extract the intrinsic parameters
fx = mtx[0, 0]  # Focal length in x
fy = mtx[1, 1]  # Focal length in y
cx = mtx[0, 2]  # Principal point x-coordinate
cy = mtx[1, 2]  # Principal point y-coordinate

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Focal lengths (fx, fy):", fx, fy)
print("Principal point (cx, cy):", cx, cy)

# Optionally, save the calibration result
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
