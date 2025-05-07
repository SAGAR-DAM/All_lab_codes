'''
author: Sagar Dam

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# cropping image

image_path="D:\\data Lab\\quantel focal spot measurement\\camera images\\Spot size_26.bmp"
image=plt.imread(image_path)
cropped_img=image[246:326,365:445]

#cropped_img=np.asarray(cropped_img)
plt.axis('off')
plt.imshow(cropped_img)
plt.show()