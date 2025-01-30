# 3D Cube
Multicamera 3D cube calibration and show


# Introduction
This repository uses two cameras and a chessboard to show a 3D cube on the chessboard.

First, you need to run the calibration.py file to calibrate the two  cameras using a printed chessboard on a flat surface. The taken images will be saved in the calib_images folder and the calibration measures in a .npz file.

Then you can run the cube.py file and show the chessboard on both of the cameras, this will draw a 3D cube on the chessboard which will remain the same i most of the angles.
