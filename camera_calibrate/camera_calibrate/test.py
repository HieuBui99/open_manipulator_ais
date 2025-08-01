import cv2
import numpy as np

import numpy as np
# import matplotlib.pyplot as plt

import pdb

class IntrinsicsEstimator:
    def __init__(self, square_size_mm, chessboard_size=(9, 6)):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        self._imgs = []

    def add(self, image):
        self._imgs.append(image)

    def get_camera_matrix(self):
        return self.camera_matrix

    def get_dist_coeffs(self):
        return self.dist_coeffs

    def calibrate(self):
        """
        Returns:
        - ret: RMS re-projection error.
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - rvecs: Rotation vectors.
        - tvecs: Translation vectors.
        """
        # Termination criteria for cornerSubPix
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # Prepare object points based on the actual chessboard dimensions
        objp = np.zeros((self.chessboard_size[1]*self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                            0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_mm  # Scale by square size to get real-world coordinates

        # Arrays to store object points and image points from all images
        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane

        # Process each image
        for idx, img in enumerate(self._imgs):
            # img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            breakpoint()
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if not ret:
                self._imgs.remove(img)
                continue 
            # If found, add object points and image points
            if ret:
                objpoints.append(objp)

                # Refine corner locations to sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners (optional)
                img_drawn = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners2, ret)
                cv2.imshow('Chessboard Detection', img_drawn)
                cv2.waitKey(5000)  # Display each image for 100ms
            else:
                print(f"Chessboard corners not found in image {fname}. Skipping.")

        cv2.destroyAllWindows()

        if not objpoints or not imgpoints:
            print("No chessboard corners were detected in any image. Calibration cannot proceed.")
            return

        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # Calculate re-projection error
        mean_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
            mean_error += error**2
            total_points += len(objpoints[i])

        mean_error = np.sqrt(mean_error / total_points)
        print(f"Calibration RMS Re-projection Error: {ret}")
        print(f"Mean Re-projection Error: {mean_error}")

        print("\nCamera Matrix:")
        print(camera_matrix)

        print("\nDistortion Coefficients:")
        print(dist_coeffs.ravel())

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def save_calibration(self, save_path):
        """
        Save the camera calibration results to a file.

        Parameters:
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - save_path: Directory to save the calibration file.
        """
        calibration_file = 'calibration_result.npz'
        np.savez(calibration_file, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
        print(f"\nCalibration results saved to {calibration_file}")


if __name__ == "__main__":
    # Example usage
    square_size_mm = 28.0  # Size of a square in millimeters
    chessboard_size = (8, 5)  # Number of internal corners per chessboard

    estimator = IntrinsicsEstimator(square_size_mm, chessboard_size)

    img = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/test.npy')
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)  
    estimator.add(img)
    estimator.calibrate()
    # estimator.save_calibration('calibration_results.npz')