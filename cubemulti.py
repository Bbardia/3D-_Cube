import cv2
import numpy as np

# Load calibration data
try:
    calibration_data = np.load('calibration_data.npz')
    mtx_left = calibration_data['mtx_left']  # Left camera matrix
    dist_left = calibration_data['dist_left']  # Left camera distortion coefficients
    mtx_right = calibration_data['mtx_right']  # Right camera matrix
    dist_right = calibration_data['dist_right']  # Right camera distortion coefficients
    R = calibration_data['R']  # Rotation matrix between left and right cameras
    T = calibration_data['T']  # Translation vector between left and right cameras
    print("Calibration data loaded successfully.")
except Exception as e:
    print("Error loading calibration data:", e)
    exit()

# Construct projection matrices for triangulation
projection_matrix_left = np.hstack((mtx_left, np.zeros((3, 1))))
extrinsic_right = np.hstack((R, T))
projection_matrix_right = mtx_right @ extrinsic_right

# Chessboard pattern size and square size
chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column
square_size = 25.0  # Size of a square in your defined unit (e.g., millimeters)

# Prepare object points based on real-world coordinates
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by the size of a square

# Initialize the cameras
left_cam = cv2.VideoCapture(0)  # Adjust the ID for the left camera as needed
right_cam = cv2.VideoCapture(1)  # Adjust the ID for the right camera as needed

# Function to draw a cube on the image
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Draw the base in red
    cv2.drawContours(img, [imgpts[:4]], -1, (0, 0, 255), 3)  # Base
    
    # Draw pillars connecting the base to the top
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)  # Vertical lines

    # Draw the top in blue
    cv2.drawContours(img, [imgpts[4:]], -1, (255, 0, 0), 3)  # Top
    return img

# Function to get 3D points from the left and right 2D points
def get_3d_points(left_points, right_points):
    # Triangulate points to get 3D coordinates using projection matrices
    points_4d_homogeneous = cv2.triangulatePoints(projection_matrix_left, projection_matrix_right, left_points.T, right_points.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T)  # Convert from homogeneous to 3D
    return points_3d

# Main loop
while True:
    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()
    if not ret_left or not ret_right:
        print("Error: Failed to capture frame from one or both cameras.")
        break

    # Convert both frames to grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in both frames
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        print("Chessboard detected in both cameras!")

        # Refine the corner locations for sub-pixel accuracy
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), 
                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Draw and display the corners in both frames
        cv2.drawChessboardCorners(left_frame, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(right_frame, chessboard_size, corners_right, ret_right)

        # Triangulate points to get the 3D position of the chessboard corners
        chessboard_3d_points = get_3d_points(corners_left, corners_right)

        # Calculate the indices of the 4 central squares
        mid_row = chessboard_size[1] // 2
        mid_col = chessboard_size[0] // 2

        # Use the 3D points corresponding to the 4 central squares
        base_3d_points = np.array([
            chessboard_3d_points[(mid_row - 1) * chessboard_size[0] + (mid_col - 1)][0],  # Top-left corner of middle square
            chessboard_3d_points[(mid_row - 1) * chessboard_size[0] + mid_col][0],  # Top-right corner of middle square
            chessboard_3d_points[mid_row * chessboard_size[0] + mid_col][0],  # Bottom-right corner of middle square
            chessboard_3d_points[mid_row * chessboard_size[0] + (mid_col - 1)][0]  # Bottom-left corner of middle square
        ], dtype="float32")

        # Use the entire chessboard's 3D points and 2D points in the left camera for pose estimation
        retval, rvec, tvec = cv2.solvePnP(objp, corners_left, mtx_left, dist_left)

        if retval:
            print(f"Rotation Vector: {rvec}\nTranslation Vector: {tvec}")

            # Define the cube's 3D coordinates using the chessboard's base and height equal to chessboard width
            cube_3d_full_points = np.array([
                base_3d_points[0], base_3d_points[1], base_3d_points[2], base_3d_points[3],  # Base corners
                base_3d_points[0] + np.array([0, 0, -square_size * chessboard_size[1] / 2]),  # Top-left corner
                base_3d_points[1] + np.array([0, 0, -square_size * chessboard_size[1] / 2]),  # Top-right corner
                base_3d_points[2] + np.array([0, 0, -square_size * chessboard_size[1] / 2]),  # Bottom-right top corner
                base_3d_points[3] + np.array([0, 0, -square_size * chessboard_size[1] / 2])  # Bottom-left top corner
            ])

            # Project the cube points onto the 2D image plane for the left camera
            projected_cube_points, _ = cv2.projectPoints(cube_3d_full_points, rvec, tvec, mtx_left, dist_left)

            # Draw the cube on the left frame
            left_frame = draw_cube(left_frame, projected_cube_points)

        else:
            print("SolvePnP failed to estimate the pose.")

    # Display both frames with the cube overlay in the left frame
    cv2.imshow("Left Camera - 3D Cube Overlay", left_frame)
    cv2.imshow("Right Camera", right_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
