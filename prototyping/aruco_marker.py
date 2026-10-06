import cv2

def generate_four_markers():
    # Use a standard 4x4 dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 200  # pixels

    for marker_id in range(4):
        # Generate the marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # Save to disk
        filename = f"marker_{marker_id}.png"
        cv2.imwrite(filename, marker_img)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    generate_four_markers()