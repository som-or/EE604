import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    final_size = (600, 600)
    resized_image = cv2.resize(image, (550, 550))
    canvas = np.zeros((final_size[0], final_size[1], 3), dtype=np.uint8)
    x_offset = (final_size[0] - resized_image.shape[0]) // 2
    y_offset = (final_size[1] - resized_image.shape[1]) // 2
    canvas[x_offset:x_offset+resized_image.shape[0], y_offset:y_offset+resized_image.shape[1]] = resized_image
    gray_img= cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    thresh=10
    binary_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]
    corners = cv2.goodFeaturesToTrack(binary_img, 4, 0.5, 100)
    corners = np.intp(corners)
    crn=corners.reshape(-1,2).tolist()

    sorted_coordinates =sorted(crn, key=lambda x: (x[1], x[0]))
    i=0
    while i<3:
        if sorted_coordinates[i][0]>sorted_coordinates[i+1][0]:
            sorted_coordinates[i],sorted_coordinates[i+1]=sorted_coordinates[i+1],sorted_coordinates[i]
        i=i+2

    pts1=np.float32(sorted_coordinates)
    pts2 = np.float32([[0,0],[600,0],[0,600],[600,600]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(canvas,M,(600,600))


    ######################################################################

    return image
