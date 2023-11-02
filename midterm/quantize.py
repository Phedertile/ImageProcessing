import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = np.array([ [220, 192, 210, 32, 58, 222, 199, 194, 213],
                 [196, 203, 220, 60, 96, 57, 193, 208, 221],
                 [62, 56, 37, 55, 115, 107, 60, 195, 217],
                 [33, 96, 108, 123, 127, 194, 98, 45, 209],
                 [49, 117, 223, 198, 213, 192, 221, 117, 51],
                 [60, 105, 99, 124, 108, 203, 126, 39, 198],
                 [55, 44, 39, 60, 97, 121, 63, 222, 211],
                 [193, 215, 207, 62, 100, 50, 194, 199, 206],
                 [204, 220, 196, 59, 47, 203, 194, 217, 202]])

for row in image:
    for pixel in row:
        print(pixel, end="\t\t")
    print()


print("-----------------------------------------------------------------------------")

quantized_image = np.floor_divide(image, 32)  # Divide by 32 to map 0-255 range to 0-7

for row in quantized_image:
    for pixel in row:
        print(pixel, end="\t\t")
    print("\n")
    
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original 8-bit image
axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original 8-Bit Image')
axes[0].axis('off')

# Quantized 3-bit image
axes[1].imshow(quantized_image, cmap='gray', vmin=0, vmax=7)
axes[1].set_title('Quantized 3-Bit Image')
axes[1].axis('off')

# Show the plots
plt.tight_layout()
plt.show()

# print(quantized_image[1][4])
# print(quantized_image[4][2])
# print(quantized_image[6][6])