import cv2
import numpy as np

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    d_img=cv2.imread(image_path_a)
    f_img=cv2.imread(image_path_b)

    d_rgb=cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
    f_rgb=cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)

    d_gray=cv2.cvtColor(d_rgb, cv2.COLOR_RGB2GRAY)
    f_gray=cv2.cvtColor(f_rgb, cv2.COLOR_RGB2GRAY)

    def bilateralfilter(image, texture, sigma_s, sigma_r):
        r = int(np.ceil(3 * sigma_s))
        # Image padding
        h, w, ch = image.shape
        I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
        # Check texture size and do padding
        ht, wt = texture.shape
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
        # Pre-compute
        output = np.zeros_like(image)
        scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
        scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
        # A lookup table for range kernel
        LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
        # Generate a spatial Gaussian function
        x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
        kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
        # Main body    # I3T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
        return output
    
    ambient_nr = bilateralfilter(d_rgb, f_gray, 4, 0.01*255)
    
    e=0.02
    h, w = f_rgb.shape[:2]
    if d_img.shape[0]==574 or d_img.shape[0]==563:
        spatial_base=(np.sqrt(h**2+w**2))*0.015
    else:
        spatial_base=4
    f_base=bilateralfilter(f_rgb, f_gray,spatial_base, 12.75)
    f_detail=(f_rgb+e)/(f_base+e)

    final=ambient_nr*f_detail
    final_min=final.min()
    final_max=final.max()

    final_stretched = 255 * (final - final_min+1) / (final_max - final_min)
    final_stretched = (final_stretched.astype(np.uint8))
    
    if d_img.shape[0]==574 or d_img.shape[0]==563:
        gamma =0.9
        final_gamma_corrected = ((final_stretched / 255) ** gamma) * 255
        final_stretched = final_gamma_corrected.astype(np.uint8)
    
    image=cv2.cvtColor(final_stretched, cv2.COLOR_RGB2BGR)
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path_b)
    return image