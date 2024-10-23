import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

N = 4 # Number of points
n = 0

# Parameters of the two images
p = np.empty((N,2))
p_flag = np.empty((N,2))

# Mouse callback function
def draw(event,x,y,flags,param):
    global n
    p = param[0]
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[1],(x,y),5,(255,0,0),-1)
        p[n] = (x,y)
        n += 1

# Importing the images and creating copies
image_background = cv.imread(r'Images/005.jpg', cv.IMREAD_COLOR)
image_superimposed = cv.imread(r"Images/flag.png", cv.IMREAD_COLOR)
image_background_copy = image_background.copy()
image_superimposed_copy = image_superimposed.copy()

# Getting the mouse points of the base image
cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
param = [p, image_background_copy]
cv.setMouseCallback('Image',draw, param)
while(1):
    cv.imshow('Image', image_background_copy)
    if n == N:
        break
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

# Automatically get the corners of the flag image
h, w = image_superimposed.shape[:2]
p_flag[0] = [0, 0]               # Top-left corner
p_flag[1] = [w - 1, 0]           # Top-right corner
p_flag[2] = [w - 1, h - 1]       # Bottom-right corner
p_flag[3] = [0, h - 1]           # Bottom-left corner


h, status = cv.findHomography(p, p_flag) # Calculating homography between image and flag

# Warping image of flag
warped_img = cv.warpPerspective(image_superimposed, np.linalg.inv(h), (image_background.shape[1],image_background.shape[0])) 

blended = cv.addWeighted(image_background, 0.5, warped_img, 0.9, 0.0)
fig, ax = plt.subplots(1,1,figsize= (8,8))
ax.imshow(cv.cvtColor(blended,cv.COLOR_BGR2RGB))

# Plotting the results
fig,ax=plt.subplots(1,3,figsize=(21,7))
ax[0].imshow(cv.cvtColor(image_background,cv.COLOR_BGR2RGB))
ax[0].set_title("Source Image")
ax[1].imshow(cv.cvtColor(image_superimposed,cv.COLOR_BGR2RGB))
ax[1].set_title("Flag Image")
ax[2].imshow(cv.cvtColor(blended,cv.COLOR_BGR2RGB))
ax[2].set_title("Final Image")