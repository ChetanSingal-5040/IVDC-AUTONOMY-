import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the road image (680 x 542)
img = cv2.imread("road.jpg")

if img is None:
    print("Image not found. Check the filename and folder.")
else:
    print("Image loaded successfully!")
    print("Image dimensions:", img.shape)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Step 2: Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection
    canny_edges = cv2.Canny(blur, 50, 150)

    height, width = gray.shape   # 542, 680

    # Step 4: Region of Interest (ROI) - the road area, below the horizon
    roi_vertices = np.array([[
        (0, 541),       # bottom-left
        (0, 435),       # left edge of road
        (400, 262),     # far left of road
        (495, 256),     # top, where the road meets the horizon
        (680, 345),     # right edge of road
        (679, 541)      # bottom-right
    ]], dtype=np.int32)

    roi_mask = np.zeros_like(canny_edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(canny_edges, roi_mask)

    # Step 5: Detect line segments using Hough Transform
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180,
                            threshold=25, minLineLength=20, maxLineGap=40)

    line_image = img_rgb.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 255, 0), 2)

    # Step 6: Final bitmap
    # White background
    lane_bitmap = np.full((height, width), 255, dtype=np.uint8)

    # Lane = road between the left road edge and the white centre line
    lane_region = np.array([[
        (0, 435),       # left road edge (image border)
        (100, 395),
        (200, 345),
        (300, 293),
        (400, 262),
        (490, 257),     # top of white line
        (445, 272),
        (425, 300),     # white line curves down...
        (430, 350),
        (445, 420),
        (470, 541),     # ...to the bottom of the image
        (0, 541)        # bottom-left
    ]], dtype=np.int32)

    # Fill the lane region BLACK
    cv2.fillPoly(lane_bitmap, lane_region, 0)

    # Save output images
    cv2.imwrite("final_lane_mask.png", lane_bitmap)
    cv2.imwrite("roi_edges.png", roi_edges)

    # Step 7: Display intermediate and final images
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(blur, cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(roi_edges, cmap="gray")
    plt.title("Canny Edges + ROI")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(line_image)
    plt.title("Hough Line Segments")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(lane_bitmap, cmap="gray", vmin=0, vmax=255)
    plt.title("Black Lane, White Background")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("pipeline_results.png", dpi=150)
    plt.show()

    print("Saved final_lane_mask.png")
    print("Saved roi_edges.png")
    print("Saved pipeline_results.png")