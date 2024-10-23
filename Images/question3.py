import cv2 as cv
import numpy as np

# Read the background and logo images
img3 = cv.imread("Images/005.jpg")
logo = cv.imread("Images/flag.png")

# Check if images are loaded correctly
if img3 is None or logo is None:
    print("Error: One or both images not found or path is incorrect.")
    exit()

# Print the shapes of the images
print(f"Background image shape: {img3.shape}")
print(f"Logo image shape: {logo.shape}")

# Store clicked points for the destination
dst_points = []

# Mouse callback function to store points
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:  # We only need 4 points
            dst_points.append((x, y))
            cv.circle(img3, (x, y), 5, (0, 0, 255), -1)  # Mark the clicked point
            cv.imshow("Select 4 Points", img3)  # Show updated image
        if len(dst_points) == 4:
            cv.destroyWindow("Select 4 Points")  # Close the window after 4 points

# Display the image and set the mouse callback
cv.imshow("Select 4 Points", img3)
cv.setMouseCallback("Select 4 Points", click_event)
cv.waitKey(0)

# Check if we have exactly 4 points
if len(dst_points) != 4:
    print("Error: You need to select exactly 4 points.")
else:
    dst_points = np.array(dst_points, dtype=np.float32)

    # Define the source points (corners of the logo)
    y, x, _ = logo.shape
    src_points = np.array([[0, y], [x, y], [x, 0], [0, 0]], dtype=np.float32)

    # Calculate the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_points, dst_points)
    print("Transformation matrix:\n", M)  # Print the matrix for debugging

    # Warp the logo to fit into the selected region in the background image
    tf_img = cv.warpPerspective(logo, M, (img3.shape[1], img3.shape[0]))

    # Display the warped logo for debugging
    cv.imshow("Warped Logo", tf_img)
    cv.waitKey(0)

    # Create a mask for the logo
    logo_gray = cv.cvtColor(tf_img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(logo_gray, 1, 255, cv.THRESH_BINARY)

    # Invert the mask to select the background region
    mask_inv = cv.bitwise_not(mask)

    # Black-out the area of the logo in the background image
    img3_bg = cv.bitwise_and(img3, img3, mask=mask_inv)

    # Take only the logo region from the warped logo image
    logo_fg = cv.bitwise_and(tf_img, tf_img, mask=mask)

    # Add the background and logo regions
    final_img = cv.add(img3_bg, logo_fg)

    # Display the final image
    cv.imshow("Final Image", final_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
