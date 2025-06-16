import cv2
import numpy as np
import os
import sys

# --- 修改点 1: 将标定点数量定义为常量 ---
NUM_CALIBRATION_POINTS = 6
TRANSFORM_MATRIX_FILE = 'transform_matrix.npy'

def calibrate_and_save(camera_index=0):
    pixel_points = []
    # --- 修改点 2: 更新窗口提示文本 ---
    window_name = f"Calibration: Click {NUM_CALIBRATION_POINTS} points, then press 'c' to confirm"

    def mouse_callback(event, x, y, flags, param):
        # --- 修改点 3: 更新鼠标回调中的逻辑判断 ---
        if event == cv2.EVENT_LBUTTONDOWN and len(pixel_points) < NUM_CALIBRATION_POINTS:
            pixel_points.append((x, y))
            print(f"Point {len(pixel_points)}/{NUM_CALIBRATION_POINTS} registered at ({x}, {y}).")

    cap = cv2.VideoCapture(camera_index)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("--- Starting Calibration ---")
    print(f"1. Click on {NUM_CALIBRATION_POINTS} reference points in the window in a specific order.")
    print(f"2. After clicking {NUM_CALIBRATION_POINTS} points, press the 'c' key to continue to coordinate input.")
    print("3. Press 'r' to reset points. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # --- 修改点 4: 更新实时文本提示 ---
        if len(pixel_points) < NUM_CALIBRATION_POINTS:
            text = f"Click point {len(pixel_points) + 1}/{NUM_CALIBRATION_POINTS}"
        else:
            text = f"{NUM_CALIBRATION_POINTS} points selected. Press 'c' to confirm."
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        for i, point in enumerate(pixel_points):
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{i+1}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return
        elif key == ord('r'):
            pixel_points = []
            print("Points reset. Please click again.")
        # --- 修改点 5: 更新确认按键的条件 ---
        elif key == ord('c') and len(pixel_points) == NUM_CALIBRATION_POINTS:
            break 

    cap.release()
    cv2.destroyAllWindows()
    
    if len(pixel_points) != NUM_CALIBRATION_POINTS:
        print("Calibration failed: Not enough points selected.")
        return

    print("\n--- Input Robot Coordinates ---")
    print(f"Please enter the robot coordinates for the {NUM_CALIBRATION_POINTS} points you clicked, in the same order.")
    robot_points = []
    # --- 修改点 6: 更新机器人坐标输入循环 ---
    for i in range(NUM_CALIBRATION_POINTS):
        while True:
            try:
                coord_str = input(f"Enter robot coords for Point {i+1} (format: x,y): ")
                x_str, y_str = coord_str.split(',')
                robot_points.append((float(x_str.strip()), float(y_str.strip())))
                break
            except ValueError:
                print("Invalid format. Please use 'x,y', e.g., 100.5,250.0")
                
    try:
        src_pts = np.float32(pixel_points)
        dst_pts = np.float32(robot_points)
        
        # --- 修改点 7: 使用 estimateAffine2D 替换 getAffineTransform ---
        # cv2.getAffineTransform 只能处理3个点.
        # cv2.estimateAffine2D 使用最小二乘法为N个点找到最佳仿射变换.
        # 它返回变换矩阵和一个 "inliers" 掩码，后者在此处可以忽略.
        transform_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts)

        if transform_matrix is None:
            print("Error: Could not compute the transformation matrix.")
            print("Please ensure the points are not all collinear and try again.")
            return

        np.save(TRANSFORM_MATRIX_FILE, transform_matrix)
        print(f"\nTransformation matrix calculated and saved to '{TRANSFORM_MATRIX_FILE}'")
        print("Matrix content:\n", transform_matrix)
        # 打印inliers可以帮助诊断哪些点可能存在较大误差
        print(f"Inliers mask (1 = good point, 0 = outlier):\n", inliers.ravel())

    except cv2.error as e:
        print(f"Error calculating transform matrix: {e}")

# =============================================================================
#  CoordinateTransformer 类和 main 函数无需任何修改
# =============================================================================

class CoordinateTransformer:
    """
    A class to handle coordinate transformations using a pre-saved matrix.
    """
    def __init__(self):
        """Loads the transformation matrix upon initialization."""
        self.matrix = None
        if os.path.exists(TRANSFORM_MATRIX_FILE):
            self.matrix = np.load(TRANSFORM_MATRIX_FILE)
            print(f"'{TRANSFORM_MATRIX_FILE}' loaded successfully.")
        else:
            print(f"Error: Transformation matrix file '{TRANSFORM_MATRIX_FILE}' not found.")
            print("Please run the `calibrate_and_save()` function first.")
            sys.exit()

    def transform_pixel_to_robot(self, pixel_coord):
        """
        Transforms a single pixel coordinate to a robot coordinate.
        :param pixel_coord: A tuple (x, y) of the pixel coordinate.
        :return: A tuple (rx, ry) of the robot coordinate.
        """
        if self.matrix is None:
            raise Exception("Transformation matrix is not loaded.")
        
        pixel_np = np.array([[pixel_coord]], dtype=np.float32)
        transformed_point = cv2.transform(pixel_np, self.matrix)
        
        return tuple(transformed_point[0][0])

    def run_interactive_mode(self, camera_index=0):
        """
        Starts an interactive session where the user can click on the camera feed
        to get real-time robot coordinates.
        """
        if self.matrix is None:
            print("Cannot run interactive mode without a loaded transformation matrix.")
            return

        window_name = "Interactive Mode (Click to get coords, 'q' to quit)"
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
            
        print("\n--- Starting Interactive Mode ---")
        print("Click anywhere in the window to see the corresponding robot coordinates.")

        # 将回调函数定义在循环外部以避免重复设置
        def interactive_mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                robot_coord = self.transform_pixel_to_robot((x, y))
                print(f"Pixel: ({x}, {y})  =>  Robot: ({robot_coord[0]:.2f}, {robot_coord[1]:.2f})")
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, interactive_mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Interactive mode finished.")

# ================== MAIN PROGRAM ==================
if __name__ == '__main__':
    print("### PART 1: CALIBRATION ###")
    calibrate_and_save()

    if not os.path.exists(TRANSFORM_MATRIX_FILE):
        print("\nCalibration did not produce a matrix file. Exiting test.")
    else:
        print("\n### PART 2: TESTING THE TRANSFORMER ###")
        transformer = CoordinateTransformer()

        print("\n--- Test Mode 1: Transform a specific pixel coordinate ---")
        test_pixel_coord = (320, 240) 
        robot_coord_mode1 = transformer.transform_pixel_to_robot(test_pixel_coord)
        print(f"Programmatically transforming pixel {test_pixel_coord}...")
        print(f"Resulting Robot Coordinate: {robot_coord_mode1}")
        
        print("\n--- Test Mode 2: Interactive clicking ---")
        transformer.run_interactive_mode()

        print("\n### TEST COMPLETE ###")