import cv2
import numpy as np
import os

def create_synthetic_board(width=800, height=600):
    board = np.ones((height, width, 3), dtype=np.uint8) * 240
    # Add light grid/texture
    cv2.rectangle(board, (20, 20), (width-20, height-20), (200, 200, 200), 2)
    
    # Load or generate 4 markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100
    
    positions = [
        (40, 40),                     # ID 0 Top-Left
        (width - 40 - marker_size, 40), # ID 1 Top-Right
        (width - 40 - marker_size, height - 40 - marker_size), # ID 2 Bottom-Right
        (40, height - 40 - marker_size)  # ID 3 Bottom-Left
    ]
    
    for marker_id, (x, y) in enumerate(positions):
        if hasattr(cv2.aruco, 'generateImageMarker'):
            m_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        else:
            m_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
        m_img_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
        board[y:y+marker_size, x:x+marker_size] = m_img_bgr
        
    return board

def draw_leds(board, led1_on=True, led2_on=True):
    img = board.copy()
    
    # LED 1: Red circle near center-left (300, 250)
    led1_center = (300, 250)
    led1_radius = 18
    if led1_on:
        # Glow effect
        cv2.circle(img, led1_center, led1_radius + 10, (100, 100, 255), -1)
        cv2.circle(img, led1_center, led1_radius + 5, (150, 150, 255), -1)
        cv2.circle(img, led1_center, led1_radius, (50, 50, 255), -1)
        cv2.circle(img, led1_center, 8, (200, 200, 255), -1) # bright core
    else:
        cv2.circle(img, led1_center, led1_radius, (50, 50, 150), -1)
        cv2.circle(img, led1_center, led1_radius, (30, 30, 100), 2)

    # LED 2: Green rectangle near center-right (520, 320)
    led2_bbox = (500, 300, 60, 40) # x, y, w, h
    x, y, w, h = led2_bbox
    if led2_on:
        cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (100, 255, 100), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 255, 50), -1)
        cv2.rectangle(img, (x+10, y+8), (x+w-10, y+h-8), (200, 255, 200), -1)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (30, 120, 30), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (20, 80, 20), 2)
        
    return img

def apply_perspective(img, angle_x=0, angle_y=0, scale=1.0, dx=0, dy=0):
    h, w = img.shape[:2]
    # Source points
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Warped target points
    d_x1 = int(w * 0.1 * angle_x)
    d_x2 = int(w * 0.1 * angle_y)
    d_y1 = int(h * 0.1 * angle_y)
    d_y2 = int(h * 0.1 * angle_x)
    
    dst_pts = np.float32([
        [0 + d_x1, 0 + d_y1],
        [w - d_x2, 0 + d_y2],
        [w - d_x1, h - d_y1],
        [0 + d_x2, h - d_y2]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), borderValue=(220, 220, 220))
    
    # Scale and translate
    if scale != 1.0 or dx != 0 or dy != 0:
        S = np.float32([[scale, 0, dx], [0, scale, dy], [0, 0, 1]])
        warped = cv2.warpPerspective(warped, S, (w, h), borderValue=(220, 220, 220))
        
    return warped

if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    board = create_synthetic_board()
    
    ref_img = draw_leds(board, led1_on=True, led2_on=True)
    cv2.imwrite("reference.png", ref_img)
    print("Saved reference.png")
    
    test_params = [
        (0.2, 0.1, 0.95, 10, 10, True, True),
        (-0.3, 0.2, 0.85, 40, 20, True, False),
        (0.4, -0.2, 1.05, -20, 0, False, True),
        (-0.2, -0.3, 0.75, 80, 50, True, True),
        (0.1, 0.4, 0.9, 0, -30, True, True),
    ]
    
    for idx, (ax, ay, sc, dx, dy, l1, l2) in enumerate(test_params, 1):
        scene = draw_leds(board, led1_on=l1, led2_on=l2)
        warped = apply_perspective(scene, ax, ay, sc, dx, dy)
        filename = f"img/image_{idx:02d}.png"
        cv2.imwrite(filename, warped)
        print(f"Saved {filename}")
