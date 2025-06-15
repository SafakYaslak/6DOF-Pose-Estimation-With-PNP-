# Pose Estimation using PnP Algorithm

This project provides a robust implementation of the Perspective-n-Point (PnP) algorithm for estimating the pose of an object in an image. Designed with computer vision enthusiasts and developers in mind, it enables users to interactively select points on an image and compute the object's pose using various PnP methods. Additionally, it explores a range of object dimensions to identify the best-fitting pose, making it a valuable tool for both educational and practical applications.

## Purpose

The primary goal of this project is to offer a practical and hands-on demonstration of the PnP algorithm within the field of computer vision. It serves as an educational resource for understanding the nuances of different PnP methods and their application in real-world scenarios. The project also showcases techniques for handling camera calibration data and undistorting images to achieve precise pose estimation, making it useful for learning and experimentation.

## Features

- **Image Loading and Undistortion**: Load images and correct lens distortion using precomputed camera calibration data.

- **Interactive Point Selection**: Use a mouse-based interface to select points on the image corresponding to object corners.

- **Multiple PnP Methods**: Compute poses using three distinct PnP approaches:
  - ITERATIVE (`cv.SOLVEPNP_ITERATIVE`)
  - IPPE (`cv.SOLVEPNP_IPPE`)
  - EPNP (`cv.SOLVEPNP_EPNP`)

- **Hybrid Pose Solution**: Combine angles from different methods (roll from ITERATIVE, pitch from EPNP, yaw from IPPE) for an optimized result.

- **Dimension Exploration**: Test various object dimensions (upper width, lower width, height) to find the best pose fit.

- **Visualization**: Display results with coordinate axes overlaid on the image and detailed angle information.

## Prerequisites

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- SciPy (`scipy`)

You can install these dependencies using pip:

```bash
pip install opencv-python numpy scipy
```

Additionally, the project requires camera calibration files (`cameraMatrix_best.pkl` and `dist1.pkl`), which are assumed to be present in the `pnp_alg` directory. These files should be generated separately through a camera calibration process (e.g., using OpenCV's calibration tools).

## Usage

Follow these steps to run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/SafakYaslak/6DOF-Pose-Estimation-With-PNP-.git
cd pnp_alg
```

### 2. Prepare Your Image

Place the image you want to analyze (e.g., `img2.png`) in the project directory or update the script with the correct path.

### 3. Verify Calibration Files

Ensure `cameraMatrix_best.pkl` and `dist1.pkl` are in the `pnp_alg` directory. These files contain the camera intrinsic matrix and distortion coefficients, respectively.

### 4. Run the Script

```bash
python 6dof_pose.py
```

### 5. Select Points

- A window will display the undistorted image.
- Left-click to select four points in the following order: **top-left, top-right, bottom-right, bottom-left**. This order is critical for accurate pose estimation.
- Right-click to undo the last point if needed.
- Press `q` or select four points to proceed.

### 6. View Results

- The script will process all dimension combinations and display the top 5 results based on their proximity to target angles (roll=0°, yaw=0°).
- Each result window shows the axes drawn on the image, along with the dimensions and angles.

### 7. Close Windows

Press any key to close each result window.

> **Note**: The script assumes the image is resized to 640x480 pixels. Adjust the `cv.resize()` call in the code if your image has a different resolution.

## Methodology

The project follows a structured approach to pose estimation:

### Camera Calibration
Utilizes precomputed camera intrinsic matrix and distortion coefficients to undistort the input image, ensuring accuracy in subsequent steps.

### Point Selection
Users manually select four points on the image, corresponding to the object's corners in a specific order.

### PnP Computation
Applies three PnP methods to estimate the pose:
- **ITERATIVE**: An iterative optimization approach
- **IPPE**: Infinitesimal Plane-Based Pose Estimation for planar objects
- **EPNP**: Efficient Perspective-n-Point for general 3D configurations

### Hybrid Solution
Constructs a hybrid pose by combining the roll angle from ITERATIVE, pitch from EPNP, and yaw from IPPE, leveraging the strengths of each method.

### Dimension Exploration
Iterates through a range of object dimensions (upper width: 16.0–18.5 cm, lower width: 21.0–23.0 cm, height: 9.8–11.0 cm) to compute poses for each combination.

### Result Selection
Ranks results by their Euclidean distance in radians to target angles (roll=0°, yaw=0°) and selects the top 5 for display.

## Interpreting Results

The output consists of the top 5 pose estimations, each presented in a separate window. For each result, you will see:

- **Dimensions**: The object's upper width, lower width, and height used for that pose (e.g., U=17.2, L=22.1, H=10.5)
- **Angles**: Roll, pitch, and yaw in degrees, with their radian equivalents and sources (e.g., Roll: 2.1° (ITERATIVE))
- **Radian Distance**: A metric indicating how close the estimated angles are to the target (smaller is better)
- **Visualization**: Coordinate axes (X=red, Y=green, Z=blue) drawn on the image, originating from the object's center

The hybrid solution aims to balance accuracy across all three angles, with the translation vector sourced from the ITERATIVE method.

## Example Output

For an input image `img2.png`, the script might produce:

```
Top 5 results:
1. Result:
   Dimensions: Upper Width=17.2, Lower Width=22.1, Height=10.5
   Angles: Roll=10.1°, Pitch=-30.8°, Yaw=0.9°
   Radian Distance: 0.412
```

The corresponding window displays the axes and detailed text overlay.

## Limitations

- Requires pre-existing calibration files; no built-in calibration tool is provided
- Assumes a planar object with four coplanar points
- Computation time increases with the number of dimension combinations tested

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request

Please report issues or suggest enhancements via the GitHub Issues tab.

## Author

Şafak Yaşlak
