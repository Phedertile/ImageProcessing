import numpy as np
import matplotlib.pyplot as plt

# Define the 2x3 pixel image (3-bit quantized)
image_2x3 = np.array([
    [2, 4, 6],
    [3, 5, 7]
], dtype=np.uint8)

# Create a new 3x4 pixel image
new_height, new_width = 3, 4
image_3x4 = np.zeros((new_height, new_width), dtype=np.uint8)

# Determine scaling factors for height and width
scale_x = image_2x3.shape[1] / image_3x4.shape[1]
scale_y = image_2x3.shape[0] / image_3x4.shape[0]

# Perform bilinear interpolation
for i in range(new_height):
    for j in range(new_width):
        # Find the corresponding coordinates in the original image
        x = j * scale_x
        y = i * scale_y

        # Calculate the four nearest neighbors
        x0 = int(x)
        x1 = x0 + 1 if x0 < image_2x3.shape[1] - 1 else x0
        y0 = int(y)
        y1 = y0 + 1 if y0 < image_2x3.shape[0] - 1 else y0

        # Bilinear interpolation
        fx = x - x0
        fy = y - y0
        interpolated_value = (1 - fx) * (1 - fy) * image_2x3[y0, x0] + \
                            fx * (1 - fy) * image_2x3[y0, x1] + \
                            (1 - fx) * fy * image_2x3[y1, x0] + \
                            fx * fy * image_2x3[y1, x1]

        # Set the interpolated value in the new image
        image_3x4[i, j] = int(round(interpolated_value))

# Visualize the original and interpolated images
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(image_2x3, cmap='gray', vmin=0, vmax=7)
plt.title('Original 2x3 Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(image_3x4, cmap='gray', vmin=0, vmax=7)
plt.title('Interpolated 3x4 Image')
plt.axis('off')

plt.tight_layout()
plt.show()
