import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)

# Calculate and display the histogram of the original image
def plot_histogram(image, title, color):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    plt.plot(bin_edges[0:-1], histogram, color=color)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

# Display original image and its histogram
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plot_histogram(image, "Histogram of Original Image", "blue")

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display equalized image and its histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plot_histogram(equalized_image, "Histogram of Equalized Image", "red")

plt.tight_layout()
plt.show()
