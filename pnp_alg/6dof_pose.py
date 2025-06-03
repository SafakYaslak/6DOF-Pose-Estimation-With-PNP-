import numpy as np
import os
import cv2 as cv
from enum import Enum
import pickle
from scipy.spatial.transform import Rotation as SciPyRotation

class DrawOption(Enum):
    AXES = 1
    CUBE = 2

# Function to draw the coordinate axes on the image
def drawAxes(img, origin, imgpts):
    """Draws the coordinate axes on the given image.

    Args:
        img (numpy.ndarray): The image on which to draw the axes.
        origin (numpy.ndarray): The origin point of the axes in image coordinates.
        imgpts (numpy.ndarray): The 3D points representing the ends of the axes projected onto the image.

    Returns:
        numpy.ndarray: The image with the coordinate axes drawn.
    """
    colors = [(0,0,255), (0,255,0), (255,0,0)]
    labels = ['X', 'Y', 'Z']
    
    origin = tuple(origin.astype(int))
    for i in range(3):
        end_point = tuple(imgpts[i].ravel().astype(int))
        img = cv.arrowedLine(img, origin, end_point, colors[i], 3, tipLength=0.1)
        text_pos = (end_point[0]+5, end_point[1]+5)
        cv.putText(img, labels[i], text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    return img

# Function to draw a cube on the image
def drawCube(img, imgpts):
    """Draws a cube on the given image.

    Args:
        img (numpy.ndarray): The image on which to draw the cube.
        imgpts (numpy.ndarray): The 3D points representing the vertices of the cube projected onto the image.

    Returns:
        numpy.ndarray: The image with the cube drawn.
    """
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    cv.drawContours(img, [imgpts[:4]], -1, (0,255,0), 3)
    for i,j in zip(range(4), range(4,8)):
        cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 3)
    cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    
    return img

selected_points = []
img_display = None

# Mouse callback function to select points on the image
def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to select points on the image.

    Args:
        event (cv2 event): The type of mouse event.
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (Any): Any extra parameters passed to the function.
    """
    global selected_points, img_display
    
    if event == cv.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv.circle(img_display, (x, y), 5, (0,0,255), -1)
        cv.imshow("Select Points", img_display)
    
    elif event == cv.EVENT_RBUTTONDOWN:
        if selected_points:
            selected_points.pop()
            img_display = param.copy()
            for pt in selected_points:
                cv.circle(img_display, tuple(pt), 5, (0,0,255), -1)
            cv.imshow("Select Points", img_display)

# Function to get user-selected points on the image
def get_user_points(img):
    """Gets user-selected points on the image.

    Args:
        img (numpy.ndarray): The image on which to select points.

    Returns:
        numpy.ndarray: An array of the selected points.
    """
    global selected_points, img_display
    selected_points = []
    img_display = img.copy()
    
    cv.imshow("Select Points", img_display)
    cv.setMouseCallback("Select Points", mouse_callback, img)
    
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or len(selected_points) >= 4:
            break
    
    cv.destroyAllWindows()
    return np.array(selected_points, dtype=np.float32)

# Function to convert a rotation matrix to Euler angles (roll, pitch, yaw)
def rotation_matrix_to_euler_angles(rotation_matrix):
    """Converts a rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        rotation_matrix (numpy.ndarray): The rotation matrix to convert.

    Returns:
        dict: A dictionary containing the roll, pitch, and yaw angles in degrees.
    """
    rot = SciPyRotation.from_matrix(rotation_matrix)
    euler_zyx = rot.as_euler('zyx', degrees=True)
    return {
        'roll': euler_zyx[2],
        'pitch': euler_zyx[1], 
        'yaw': euler_zyx[0]
    }

# Function to draw the PnP result on the image
def draw_result(img, rvec, tvec, camera_matrix, dist_coeff, method_name, angles):
    """Draws the PnP result on the image.

    Args:
        img (numpy.ndarray): The image on which to draw the result.
        rvec (numpy.ndarray): The rotation vector.
        tvec (numpy.ndarray): The translation vector.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeff (numpy.ndarray): The distortion coefficients.
        method_name (str): The name of the PnP method used.
        angles (dict): A dictionary containing the roll, pitch, and yaw angles.

    Returns:
        numpy.ndarray: The image with the PnP result drawn.
    """
    axis_length = 10
    axis_points = np.float32([
        [axis_length,0,0],
        [0,axis_length,0],
        [0,0,-axis_length]
    ]).reshape(-1,3)
    
    origin_3d = np.zeros((1,3), dtype=np.float32)
    origin_2d, _ = cv.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeff)
    origin = origin_2d[0][0]
    
    imgpts, _ = cv.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeff)
    img_out = drawAxes(img.copy(), origin, imgpts)
    
    text = [
        method_name,
        f"Roll: {angles['roll']:.1f}° ({np.deg2rad(angles['roll']):.3f} rad)",
        f"Pitch: {angles['pitch']:.1f}° ({np.deg2rad(angles['pitch']):.3f} rad)",
        f"Yaw: {angles['yaw']:.1f}° ({np.deg2rad(angles['yaw']):.3f} rad)"
    ]
    
    y_start = 40
    for line in text:
        cv.putText(img_out, line, (20, y_start), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv.putText(img_out, line, (20, y_start), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_start += 35
    
    return img_out

# Function to draw the hybrid PnP result on the image
def draw_hybrid_result(img, rvec, tvec, camera_matrix, dist_coeff, angles, sources, dimensions):
    """Draws the hybrid PnP result on the image.

    Args:
        img (numpy.ndarray): The image on which to draw the result.
        rvec (numpy.ndarray): The rotation vector.
        tvec (numpy.ndarray): The translation vector.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeff (numpy.ndarray): The distortion coefficients.
        angles (dict): A dictionary containing the roll, pitch, and yaw angles.
        sources (dict): A dictionary containing the sources of the roll, pitch, and yaw angles.
        dimensions (tuple): The dimensions of the object.

    Returns:
        numpy.ndarray: The image with the hybrid PnP result drawn.
    """
    axis_length = 10
    axis_points = np.float32([
        [axis_length,0,0],
        [0,axis_length,0],
        [0,0,-axis_length]
    ]).reshape(-1,3)
    
    origin_3d = np.zeros((1,3), dtype=np.float32)
    origin_2d, _ = cv.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeff)
    origin = origin_2d[0][0]
    
    imgpts, _ = cv.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeff)
    img_out = drawAxes(img.copy(), origin, imgpts)
    
    text = [
        "HYBRID SOLUTION",
        f"Dimensions: U={dimensions[0]:.1f}, L={dimensions[1]:.1f}",
        f"Roll: {angles['roll']:.1f}° ({sources['Roll']}) ---- ({np.deg2rad(angles['roll']):.3f} rad)",
       
        f"Yaw: {angles['yaw']:.1f}° ({sources['Yaw']}) ---- ({np.deg2rad(angles['yaw']):.3f} rad)",
      
     
        f"Translation: {tvec.squeeze().round(2)} (ITERATIVE)"
    ]
    
    y_start = 40
    for line in text:
        cv.putText(img_out, line, (20, y_start), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv.putText(img_out, line, (20, y_start), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_start += 35
    
    return img_out

# Function to compute PnP methods
def compute_pnp_methods(img, world_points, img_points, camMatrix, distCoeff):
    """Computes the PnP (Perspective-n-Point) pose estimation using different methods.

    Args:
        img (numpy.ndarray): The input image.
        world_points (numpy.ndarray): The 3D coordinates of the object points in the world coordinate system.
        img_points (numpy.ndarray): The 2D coordinates of the object points in the image.
        camMatrix (numpy.ndarray): The camera intrinsic matrix.
        distCoeff (numpy.ndarray): The camera distortion coefficients.

    Returns:
        dict: A dictionary containing the results of the hybrid PnP method.
    """
    methods = [
        ('ITERATIVE', cv.SOLVEPNP_ITERATIVE),
        ('IPPE', cv.SOLVEPNP_IPPE),
        ('EPNP', cv.SOLVEPNP_EPNP)
    ]
    
    method_data = {}
    
    for method_name, flag in methods:
        try:
            success, rvec, tvec = cv.solvePnP(
                world_points,
                img_points,
                camMatrix,
                distCoeff,
                flags=flag
            )
            
            if success:
                R, _ = cv.Rodrigues(rvec)
                angles = rotation_matrix_to_euler_angles(R)
                method_data[method_name] = {
                    'rvec': rvec,
                    'tvec': tvec,
                    'angles': angles
                }
        except:
            continue
    
    hybrid_data = None
    if len(method_data) >= 3:
        try:
            hybrid_angles = {
                'roll': method_data['ITERATIVE']['angles']['roll'],
                'pitch': method_data['EPNP']['angles']['pitch'],
                'yaw': method_data['IPPE']['angles']['yaw']
            }
            
            rot = SciPyRotation.from_euler('zyx', [
                hybrid_angles['yaw'],
                hybrid_angles['pitch'],
                hybrid_angles['roll']
            ], degrees=True)
            
            R_hybrid = rot.as_matrix()
            rvec_hybrid, _ = cv.Rodrigues(R_hybrid)
            tvec_hybrid = method_data['ITERATIVE']['tvec']
            
            hybrid_data = {
                'rvec': rvec_hybrid,
                'tvec': tvec_hybrid,
                'angles': hybrid_angles,
                'sources': {'Roll': 'ITERATIVE', 'Pitch': 'EPNP', 'Yaw': 'IPPE'}
            }
        except:
            pass
    
    return hybrid_data

# Main function for pose estimation
def poseEstimation(img_path):
    """Estimates the pose of an object in an image using the PnP method.

    Args:
        img_path (str): The path to the image.
    """
    # Load calibration data
    try:
        with open(r"pnp_alg\cameraMatrix_best.pkl", "rb") as f:
            camMatrix = pickle.load(f)
        with open(r"pnp_alg\dist1.pkl", "rb") as f:
            distCoeff = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Calibration files not found!")

    # Load and undistort the image
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv.resize(img, (640, 480))
    img_undistorted = cv.undistort(img, camMatrix, distCoeff)
    
    # Get user-selected points
    print("Select 4 points: top-left -> top-right -> bottom-right -> bottom-left")
    img_points = get_user_points(img_undistorted)
    
    if len(img_points) != 4:
        raise ValueError("You must select exactly 4 points!")

    # Parameter ranges
    upper_widths = np.arange(16.0, 18.6, 0.1)
    lower_widths = np.arange(21.0, 23.1, 0.1)
    heights = np.arange(9.8, 11.1, 0.1)

    results = []
    total_combinations = len(upper_widths) * len(lower_widths) * len(heights)
    current = 0

    # Iterate through all combinations to find the best 5 results
    for upper_w in upper_widths:
        for lower_w in lower_widths:
            for height in heights:
                current += 1
                print(f"Progress: {current}/{total_combinations} ({(current/total_combinations)*100:.1f}%)")
                
                world_points = np.array([
                    [-upper_w/2, -height/2, 0],
                    [upper_w/2, -height/2, 0],
                    [lower_w/2, height/2, 0],
                    [-lower_w/2, height/2, 0]
                ], dtype=np.float32)

                hybrid_data = compute_pnp_methods(
                    img_undistorted, 
                    world_points, 
                    img_points, 
                    camMatrix, 
                    distCoeff
                )

                if hybrid_data:
                    # Angles in radians
                    roll_rad = np.deg2rad(hybrid_data['angles']['roll'])
                    pitch_rad = np.deg2rad(hybrid_data['angles']['pitch'])
                    yaw_rad = np.deg2rad(hybrid_data['angles']['yaw'])
                    
                    # Distance to target values
                    distance = np.sqrt(
                        (roll_rad -0)**2 +
                       
                        (yaw_rad - 0)**2
                    )
          
                    results.append((
                        distance,
                        (round(upper_w,1),  round(height,1)),
                        hybrid_data
                    ))
    
    # Select the top 5 results
    results.sort(key=lambda x: x[0])
    top_5 = results[:5]

    # Display the results
    print("\nTop 5 results:")
    for idx, (distance, dimensions, data) in enumerate(top_5, 1):
        print(f"\n{idx}. Result:")
        print(f"Dimensions: Upper Width={dimensions[0]}, Lower Width={dimensions[1]}")
        print(f"Angles: Roll={data['angles']['roll']:.1f}°, Pitch={data['angles']['pitch']:.1f}°, Yaw={data['angles']['yaw']:.1f}°")
        print(f"Radian Distance: {distance:.4f}")
        
        # Visualize the hybrid result
        result_img = draw_hybrid_result(
            img_undistorted.copy(),
            data['rvec'],
            data['tvec'],
            camMatrix,
            distCoeff,
            data['angles'],
            data['sources'],
            dimensions
        )
        
        cv.imshow(f"Top {idx} - Dimensions: {dimensions}", result_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == '__main__':
    poseEstimation(r"pnp_alg\img2.png")
