import numpy as np
import matplotlib.pyplot as plt

# Define the 3x3 median filter kernel
median_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

# Define a sample 3-bit quantized image (replace with your own image data)
image_3bit = np.array([
    [6, 6, 6, 6, 6, 6, 6 ,6 ,6],
    [6, 2, 6, 1, 3, 1, 6, 6, 6],
    [6, 6, 6, 1, 1, 1, 6, 6, 6],
    [6, 1, 1, 7, 1, 1, 1, 1, 6],
    [6, 1, 1, 1, 1, 6, 1, 5, 6],
    [6, 1, 6, 1, 1, 1, 1, 1, 6],
    [6, 6, 6, 1, 1, 1, 6, 6, 6],
    [6, 6, 6, 1, 1, 1, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6, 6, 6]
], dtype=np.uint8)

# Function to apply the 3x3 median filter
def apply_median_filter(image, kernel):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    pad = kernel.shape[0] // 2  # Padding for boundary pixels

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the neighborhood
            neighborhood = image[i - pad:i + pad + 1, j - pad:j + pad + 1]

            # Apply median filter to the neighborhood
            median_value = np.median(neighborhood)

            # Set the median value in the filtered image
            filtered_image[i, j] = int(median_value)

    return filtered_image

# Apply the median filter to the 3-bit image
filtered_image = apply_median_filter(image_3bit, median_kernel)

print(filtered_image)
# Visualize the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_3bit, cmap='gray', vmin=0, vmax=7)
plt.title('Original 3-Bit Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray', vmin=0, vmax=7)
plt.title('Filtered Image (3x3 Median Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()
