import cv2
import numpy as np

def solution(audio_path):
    
    trial=cv2.imread(audio_path)

    ref1_features=np.array([ 6.15225352e+04, -7.29026062e+02, -1.61302686e+04,  1.07397827e+03,
       -1.97663340e+04, -4.16203705e+02, -1.15954912e+04,  9.64008026e+01,
       -6.02540039e+03,  1.92693115e+02, -6.01492249e+02,  3.44478058e+02,
       -1.11295813e+03, -5.83022522e+02,  2.21425366e+03,  4.49220001e+02,
       -3.65619507e+02, -1.13333275e+02, -4.10323853e+02, -5.40902954e+02,
       -1.12634365e+04,  3.31690178e+01,  4.08592261e+03,  1.32051193e+02,
        1.63780383e+03, -3.38071106e+02,  2.52590161e+03,  1.61867874e+02,
        1.19272681e+03, -9.71918106e+01,  1.17681018e+03, -6.43176331e+02,
        6.66228943e+02,  6.98511719e+02, -1.91015491e+03, -2.25932663e+02,
        4.12037506e+01,  2.39248398e+02, -2.45127316e+01,  2.11332336e+02,
       -2.33829141e+04,  7.78772095e+02,  4.23604004e+03, -1.00363531e+03,
        9.64256348e+03,  3.98453857e+02,  4.58201904e+03, -2.08816422e+02,
        2.44885278e+03, -1.71393845e+02,  2.83972412e+03,  5.32455444e+02,
       -1.34496094e+03, -1.59635056e+02, -1.29816858e+03, -5.34717834e+02,
        6.21664124e+02,  4.88931274e+00,  3.59655518e+02,  2.93434387e+02,
       -1.52859189e+04, -5.60713562e+02,  4.44139014e+03,  1.49810120e+02,
        5.27351367e+03,  4.51661255e+02,  2.30558496e+03,  4.38527756e+01,
        1.07289380e+03,  2.18693451e+02, -6.05819385e+03,  1.05920067e+01,
        2.42627124e+03, -2.79793976e+02,  2.13124316e+03,  3.69393433e+02,
       -9.88914204e+00, -1.78896011e+02,  5.10263367e+02,  2.22081421e+02,
       -3.31657178e+03,  6.52875183e+02,  1.23175745e+03, -4.23943542e+02,
        7.88825562e+02, -1.50329681e+02,  5.14393616e+02, -1.49583374e+02,
        4.08467590e+02, -1.65812729e+02,  2.47131885e+03, -8.32197266e+02,
        5.99629700e+02,  8.30792236e+02, -1.90606165e+03,  9.00746536e+01,
       -7.59240601e+02, -2.83802719e+01, -5.75078674e+02,  6.09292412e+01]).astype(np.float32)
    
    ref2_features=np.array([ 5.45198242e+04, -8.90390076e+02, -1.07314971e+04,  1.37497986e+03,
       -2.06167930e+04, -3.21619324e+02, -1.16078076e+04, -6.67280701e+02,
       -4.38735400e+03,  1.14807153e+03, -1.39818542e+03,  4.53567291e+02,
       -3.33860083e+03, -9.09202393e+02,  4.44668311e+03,  3.61771301e+02,
        1.19605957e+03,  6.56419556e+02, -9.80994324e+02, -9.88519104e+02,
       -6.37618018e+03,  1.60793945e+02,  3.42424255e+02, -1.07043152e+02,
        2.71626172e+03, -5.49390678e+01,  1.17349292e+03, -1.33805496e+02,
        1.47869226e+03, -3.62073860e+01,  6.37518848e+03, -4.51265564e+02,
        1.46315955e+03,  7.27047302e+02, -5.85694678e+03, -2.95447510e+02,
       -1.39340564e+03, -2.50383560e+02, -3.75347786e+01,  7.11423889e+02,
       -2.59777324e+04,  3.19049347e+02,  5.88178223e+03, -6.73412170e+02,
        9.73776367e+03,  2.80944794e+02,  5.49887891e+03,  5.38812561e+02,
        1.05736816e+03, -8.54150818e+02, -3.57329590e+03,  2.75244629e+02,
       -2.84624920e+01,  1.40956335e+01,  1.98251160e+03, -2.16010315e+02,
        4.18050812e+02, -4.57855042e+02,  1.34832495e+03,  3.50827728e+02,
       -9.01319922e+03,  3.25392670e+02,  1.92164172e+03, -3.17352478e+02,
        2.78861914e+03, -6.72545013e+01,  2.64867212e+03,  9.86981964e+01,
        7.19757385e+02,  1.02745008e+01, -2.92955396e+03, -4.62326141e+02,
        3.44877051e+03,  3.71667603e+02, -8.01520813e+02,  1.89360352e+02,
       -5.05675476e+02, -9.11422424e+01, -2.20559921e+02,  4.46793213e+01,
       -9.01806348e+03,  7.00651016e+01,  1.11215698e+03, -3.15210510e+02,
        4.74898877e+03,  1.88122162e+02,  1.42410840e+03,  2.07229736e+02,
        6.88617737e+02, -1.98055084e+02,  2.41630905e+02,  1.57133820e+02,
       -4.24315582e+02, -6.44350891e+01, -3.78824402e+02, -1.32504745e+02,
        5.21918823e+02, -5.47837067e+00,  2.45705902e+02,  3.30952415e+01]).astype(np.float32)

    def resize(img):
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        gray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out small contours based on area
        # min_contour_area = 10000 # Adjust this threshold based on your image
        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(box)
        resized_object = cv2.resize(img[y:y+h, x:x+w], (400, 400))
        H=resized_object.shape[0]
        W=resized_object.shape[1]
        # Calculate the center of the bounding box
        # Create a new canvas
        new_img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Calculate the offset to center the object in the new canvas
        offset_x = (500 - W) // 2
        offset_y = (500 - H) // 2

        # Copy the object to the center of the new canvas
        new_img[offset_y:offset_y + H, offset_x:offset_x + W] = resized_object
        
        return new_img
    
    trial=resize(trial)

    def dct_features(img):
    # Apply 2D DCT    
        img_float32 = np.float32(img)
        img_float32=cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
        # Apply 2D DCT
        dct_result = cv2.dct(img_float32)

        # Select a region of interest (ROI) based on the DCT coefficients
        roi = dct_result[:10, :10]  # Adjust the size of the ROI as needed

        # Extract features from the ROI
        features = np.ravel(roi)

        return features

    def calculate_mse(features1, features2):
        return np.mean((features1 - features2) ** 2)
    
    trial_features = dct_features(trial)

    mse1 = calculate_mse(ref1_features, trial_features)
    mse2 = calculate_mse(ref2_features, trial_features)
    threshold = 10

    if mse1 < threshold or mse2 < threshold:
        class_name="real"
    else:
        class_name="fake"

    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'fake'
    return class_name
