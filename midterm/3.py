

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread("assignment2_image1.jpg", cv.IMREAD_GRAYSCALE)  # Read as grayscale
image = np.array([[]])
width, height = image.shape
n = width * height

"""
image.flatten() returns 1 array dimension of image
bins is interval that we seperate to count group of values
ex. bins = 8 -> [frequency of 0-7], [frequency of 8-15], [frequency of 16-23],...
in this case we want to count every value that appear in image
so we set bins = 256 to count every value one by one
range = considering data, ignore if not in range
"""
# Calculate the histogram of the image
hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

# downscale with its size to caculate CDF
hist_downscaled = hist / n

# Display histogram
plt.figure(figsize=(8, 6))
plt.title('Histogram of Original')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.bar(np.arange(256), hist_downscaled, color='gray')
plt.show()

# Calculate the cumulative distribution function (CDF) of the histogram
cdf = hist_downscaled.cumsum()

# upscale back to replace original image
cdf_upscaled = cdf * 256

"""
Apply histogram equalization to the image using the lookup table (its numpy function and can't do with normal array)
it will map each value pixel with value in cdf
ex. pixel of image with value = 252 will be mapped with value of index at 252 in cdf
"""

'''
More example :

a = np.array([0,3,2,6,42,7])
b= np.array([[5,5,5,5,5],[1,1,1,1,1]])

print(a[b]) -> [[7 7 7 7 7], [3 3 3 3 3]]
'''

# equivalent with "equalized_image = cdf[image]"
equalized_image = np.take(cdf_upscaled, image)

cv.imwrite('./image/output_global.jpg', equalized_image)

display(HTML('<h3>Enhanced image</h3>'))
display(Image("./image/output_global.jpg") )
