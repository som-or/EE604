import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image = cv2.imread(image_path)
    
    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_img, 140, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)
    invert_blur_image = cv2.GaussianBlur(inverted_image, (15, 15), 0)

    contours, _ = cv2.findContours(invert_blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = np.ones_like(binary_image) * 255    
    asss=cv2.drawContours(output_image, contours, 0, (0, 255, 0), thickness=1)
    asss=cv2.bitwise_not(asss)

    lines = cv2.HoughLines(asss, 1, np.pi / 180, threshold=50)
    rho,theta = lines[0][0]
    angle = np.degrees(theta) - 90

    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_height, rotated_width = image.shape[0], image.shape[1]
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    rotated_height = int(image.shape[1] * sin_theta + image.shape[0] * cos_theta)
    rotated_width = int(image.shape[1] * cos_theta + image.shape[0] * sin_theta)

    rotation_matrix[0, 2] += (rotated_width - image.shape[1]) / 2
    rotation_matrix[1, 2] += (rotated_height - image.shape[0]) / 2

    rotated_image = cv2.warpAffine(image, rotation_matrix, (rotated_width, rotated_height))

    if abs(angle)!=90:
        c_x=int(image.shape[1]/2)
        c_y=int(image.shape[0]/2)

        ecy=int((rho -c_x* np.cos(theta)) / np.sin(theta))

        if c_y<ecy:
            rotated_image90 = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image180 = cv2.rotate(rotated_image90, cv2.ROTATE_90_CLOCKWISE)
            final=rotated_image180
        else:
            final=rotated_image
    
    image=final


    return image
