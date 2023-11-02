import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image reading
# Load the image (replace 'your_image.jpg' with your image file path)
image = cv2.imread('assignment2_image1.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the image.")
    exit()
# Perform 8-bit to 3-bit quantization
quantized_image = np.floor_divide(image, 32)  # Divide by 32 to map 0-255 range to 0-7
# Create subplots to display the original and quantized images
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
