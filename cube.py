import cv2
import numpy as np

# Load calibration data
data = np.load('calibration_data.npz')
mtx_left = data['mtx_left']
dist_left = data['dist_left']
mtx_right = data['mtx_right']
dist_right = data['dist_right']

# Parameters
chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column
square_size = 25.0        # Size of a square in your defined unit (e.g., millimeters)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on real-world coordinates
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# Define the big cube's 3D points in world coordinates
cube_size = square_size * 3  # Making the cube larger
cube_points = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, -cube_size],
    [cube_size, 0, -cube_size],
    [cube_size, cube_size, -cube_size],
    [0, cube_size, -cube_size]
], dtype=np.float32)

# Start video capture from both cameras
cap_left = cv2.VideoCapture(0)  # Adjust the index as needed
cap_right = cv2.VideoCapture(1) # Adjust the index as needed

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Failed to open cameras")
    cap_left.release()
    cap_right.release()
    exit()

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        print("Failed to capture frames")
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in both images
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left:
        # Refine corner locations for left camera
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)

        # Compute rotation and translation vectors using solvePnP for left camera
        retval_left, rvecs_left, tvecs_left = cv2.solvePnP(objp, corners_left, mtx_left, dist_left)

        # Project 3D points to image plane for left camera
        imgpts_left, _ = cv2.projectPoints(cube_points, rvecs_left, tvecs_left, mtx_left, dist_left)
        imgpts_left = np.int32(imgpts_left).reshape(-1, 2)

        # Draw the cube on the left image
        # Draw base in green
        cv2.drawContours(frame_left, [imgpts_left[:4]], -1, (0, 255, 0), 3)
        # Draw pillars in blue
        for i in range(4):
            cv2.line(frame_left, tuple(imgpts_left[i]), tuple(imgpts_left[i + 4]), (255, 0, 0), 3)
        # Draw top in red
        cv2.drawContours(frame_left, [imgpts_left[4:]], -1, (0, 0, 255), 3)

    if ret_right:
        # Refine corner locations for right camera
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        # Compute rotation and translation vectors using solvePnP for right camera
        retval_right, rvecs_right, tvecs_right = cv2.solvePnP(objp, corners_right, mtx_right, dist_right)

        # Project 3D points to image plane for right camera
        imgpts_right, _ = cv2.projectPoints(cube_points, rvecs_right, tvecs_right, mtx_right, dist_right)
        imgpts_right = np.int32(imgpts_right).reshape(-1, 2)

        # Draw the cube on the right image
        # Draw base in green
        cv2.drawContours(frame_right, [imgpts_right[:4]], -1, (0, 255, 0), 3)
        # Draw pillars in blue
        for i in range(4):
            cv2.line(frame_right, tuple(imgpts_right[i]), tuple(imgpts_right[i + 4]), (255, 0, 0), 3)
        # Draw top in red
        cv2.drawContours(frame_right, [imgpts_right[4:]], -1, (0, 0, 255), 3)

    # Display the images
    cv2.imshow('Left Camera with Cube', frame_left)
    cv2.imshow('Right Camera with Cube', frame_right)

    # Break the loop when 'q' is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
