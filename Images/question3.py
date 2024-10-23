import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

# Load the architectural image and the flag image
arch_image = cv.imread('Images/your_architectural_image.jpg')
flag_image = cv.imread('Images/your_flag_image.png')

# Check if images are loaded correctly
if arch_image is None or flag_image is None:
    print("Error: One or both images not found or path is incorrect.")
    exit()

# Store clicked points for the destination
dst_points = []

# Mouse callback function to store points
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:  # We only need 4 points
            dst_points.append((x, y))
            cv.circle(arch_image, (x, y), 5, (0, 0, 255), -1)
            print(f"Point {len(dst_points)}: ({x}, {y})")  # Print point for debugging
            cv.imshow("Select 4 Points", arch_image)
        if len(dst_points) == 4:
            cv.destroyWindow("Select 4 Points")

# Display the image and set the mouse callback
cv.imshow("Select 4 Points", arch_image)
cv.setMouseCallback("Select 4 Points", click_event)
cv.waitKey(0)

# Check if we have exactly 4 points
if len(dst_points) != 4:
    print("Error: You need to select exactly 4 points.")
    exit()

# Convert destination points to numpy array
dst_points = np.array(dst_points, dtype=np.float32)

# Define the source points (corners of the flag)
h, w = flag_image.shape[:2]
src_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

# Calculate the perspective transformation matrix
M = cv.getPerspectiveTransform(src_points, dst_points)
print("Transformation matrix:\n", M)  # Print matrix for debugging

# Warp the flag to fit into the selected region in the architectural image
warped_flag = cv.warpPerspective(flag_image, M, (arch_image.shape[1], arch_image.shape[0]))

# Create a mask for blending based on the non-black pixels in the transformed flag
flag_gray = cv.cvtColor(warped_flag, cv.COLOR_BGR2GRAY)
_, mask = cv.threshold(flag_gray, 1, 255, cv.THRESH_BINARY)

# Invert the mask to select the background region
mask_inv = cv.bitwise_not(mask)

# Black-out the area of the flag in the architectural image
arch_bg = cv.bitwise_and(arch_image, arch_image, mask=mask_inv)

# Take only the flag region from the transformed image
flag_fg = cv.bitwise_and(warped_flag, warped_flag, mask=mask)

# Add the background and flag regions
final_image = cv.add(arch_bg, flag_fg)

# Display the final image
cv.imshow("Final Image", final_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Plotting the results
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(cv.cvtColor(arch_image, cv.COLOR_BGR2RGB))
ax[0].set_title("Architectural Image")
ax[1].imshow(cv.cvtColor(flag_image, cv.COLOR_BGR2RGB))
ax[1].set_title("Flag Image")
ax[2].imshow(cv.cvtColor(final_image, cv.COLOR_BGR2RGB))
ax[2].set_title("Final Image")
plt.show()