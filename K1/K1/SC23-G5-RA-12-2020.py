import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

y_pred = []
y_true = [3,9,6,8,10,2,5,4,4,10]

for i in range(1,11):

    image = cv2.cvtColor(cv2.imread('pictures2/' + f'picture_{i}.jpg'), cv2.COLOR_BGR2RGB)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image_higher_brightness = cv2.convertScaleAbs(image_gray, beta = 20)
    image_lower_contrast = cv2.convertScaleAbs(image_higher_brightness, alpha=0.5)

    ret, image_bin = cv2.threshold(image_lower_contrast, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    image_eroded = cv2.erode(image_bin, kernel, iterations=3)
    image_dilated = cv2.dilate(image_eroded, kernel, iterations=7)

    contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_bulbasaurs = [] 

    for contour in contours: 
        center, size, angle = cv2.minAreaRect(contour) 
        height, width = size

        if width > 40 and width < 120 and height > 30 and height < 110: 
            contours_bulbasaurs.append(contour) 

    img = image.copy()
    cv2.drawContours(img, contours_bulbasaurs, -1, (255, 0, 0), 1)
    #plt.imshow(img)
    
    print(f"picture_{i}.jpg-{y_true[i-1]}-{len(contours_bulbasaurs)}")

    y_pred.append(len(contours_bulbasaurs))
    #plt.imshow(image_dilated, "gray")
    #plt.show()
mae = mean_absolute_error(y_true,y_pred)
print('MAE: ',mae)
