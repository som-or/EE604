import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_lab=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,B = cv2.split(image_lab)

    def EM_segmentation(img):
        em = cv2.ml.EM_create()

    # Load your data or create a sample dataset
        data = img.reshape(-1,1) # Replace with your actual data

        # Set the number of clusters
        num_clusters = 2
        em.setClustersNumber(num_clusters)

        # Optionally, set other parameters like the termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 150, 0.1)
        em.setTermCriteria(criteria)

        # Train the EM algorithm
        retval, log_likelihoods, labels, probs = em.trainEM(data)

        # Get the results
        means = em.getMeans()

        def BW(data):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    a=data[i][j]
                    data[i][j]=means[a]
            return data

        shape=img.shape
        labels=labels.reshape(shape)
        imgem=BW(labels)
        imgem = imgem.astype(np.uint8)
        output=cv2.medianBlur(imgem, 11) 

        return output
    
    em_B=EM_segmentation(B)

    def cotouring(img):
        _, binary_image = cv2.threshold(img, 0,255,cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours based on area
        min_contour_area = 10000 # Adjust this threshold based on your image
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        max_contour = max(contours, key=cv2.contourArea)
        # Create a mask and fill the large black spots with white
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        return mask

        # Invert the mask and combine it with the original image
        # result = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    final=cotouring(em_B)
    final_gray=cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
    _,image=cv2.threshold(final_gray, 0,255,cv2.THRESH_OTSU)

    image=cv2.merge([image,image,image])


    ######################################################################  
    return image
