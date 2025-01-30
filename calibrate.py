import cv2
import numpy as np
import glob

# Parameters
chessboard_size = (8, 6)  # Number of inner corners per a chessboard row and column
square_size = 25.0  # Size of a square in your defined unit (e.g., meters)

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare object points based on real world coordinates
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera
imgpoints_right = []  # 2d points in image plane for right camera

# Capture calibration images from both cameras
def capture_calibration_images():
    cap_left = cv2.VideoCapture(0)  # Adjust device indices as needed
    cap_right = cv2.VideoCapture(1)
    count = 0

    while count < 20:  # Capture 20 image pairs
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        cv2.imshow('Left Camera', frame_left)
        cv2.imshow('Right Camera', frame_right)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            cv2.imwrite(f'calib_images/left_{count}.png', frame_left)
            cv2.imwrite(f'calib_images/right_{count}.png', frame_right)
            print(f'Captured image pair {count}')
            count += 1
        elif key & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

# Uncomment to capture calibration images
capture_calibration_images()

# Load images
images_left = glob.glob('calib_images/left_*.png')
images_right = glob.glob('calib_images/right_*.png')

for fname_left, fname_right in zip(images_left, images_right):
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        objpoints.append(objp)

        # Refine corner locations
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
        cv2.imshow('Left Camera', img_left)
        cv2.imshow('Right Camera', img_right)
        cv2.waitKey(500)
    else:
        print(f"Chessboard not detected in pair {fname_left} and {fname_right}")

cv2.destroyAllWindows()

# Calibrate individual cameras
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left,
    mtx_right, dist_right,
    gray_left.shape[::-1],
    criteria=criteria_stereo, flags=flags)

# Save calibration results
np.savez('calibration_data.npz',
         mtx_left=mtx_left, dist_left=dist_left,
         mtx_right=mtx_right, dist_right=dist_right,
         R=R, T=T, E=E, F=F)

print("Calibration completed and saved to 'calibration_data.npz'")
