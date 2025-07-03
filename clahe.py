import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

def histogramEqual():
    # Load original image in grayscale
    root = os.getcwd()
    imgPath = os.path.join(root, 'bad_image_cat.ppm')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # Make sure image loaded
    if img is None:
        print("Image not found!")
        return

    # Create a low contrast version
    # multiplyping with 0.2 to reduce the rgb val therefore increase darkness
    bad_img = np.clip(img * 0.2 + 15, 0, 255).astype(np.uint8)

    # Equalize the bad image
    equImg = cv.equalizeHist(bad_img)

    # Histograms and CDFs
    hist = cv.calcHist([bad_img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * hist.max() / cdf.max()

    equHist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    equCdf = equHist.cumsum()
    equCdfNorm = equCdf * equHist.max() / equCdf.max()

    
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 2, 1)
    plt.imshow(bad_img, cmap='gray')
    plt.title("Bad Image (Low Contrast)")
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(equImg, cmap='gray')
    plt.title("Equalized Image")
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.plot(hist, color='red', label='Histogram')
    plt.plot(cdfNorm, color='blue', label='CDF')
    plt.title("Bad Image Histogram + CDF")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(equHist, color='red', label='Histogram')
    plt.plot(equCdfNorm, color='blue', label='CDF')
    plt.title("Equalized Histogram + CDF")
    plt.legend()
    
    # creating the clahe enhanced image taking 8 * 8 tile
    
    claheObj = cv.createCLAHE(clipLimit=5,tileGridSize=(8,8))
    claheImg = claheObj.apply(bad_img)
    claheHist = cv.calcHist([claheImg],[0],None,[256],[0,256])
    claheCdf = claheHist.cumsum()
    claheCdfNorm = claheCdf* float(claheHist.max())/claheCdf.max()
    
    plt.subplot(3,2,5)
    plt.imshow(claheImg,cmap='gray')
    plt.subplot(3,2,6)
    plt.plot(claheHist)
    plt.plot(claheCdfNorm,color='b')
    

    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == '__main__':
    histogramEqual()
