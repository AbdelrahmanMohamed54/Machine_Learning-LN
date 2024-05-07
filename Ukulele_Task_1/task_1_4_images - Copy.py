import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load and Display Original Image
original_image = cv2.imread('ukulele.jpg')
plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


# Display Separate Color Channels as Grayscale Images
channels = cv2.split(original_image)
channel_names = ['Red', 'Green', 'Blue']
for i, channel in enumerate(channels):
    plt.subplot(2, 2, i+2)
    plt.imshow(channel, cmap='gray')
    plt.title(channel_names[i] + ' Channel')

plt.show()

# Store Image as Grayscale
green_channel = channels[1]
cv2.imwrite('ukulele_grayscale.jpg', green_channel)

# Compute Mean of Green Channel
mean_green = np.mean(green_channel)

# Compute Fraction of Pixels with Value Smaller than 50 in Red Channel
red_pixels_lt_50 = np.count_nonzero(channels[2] < 50)
fraction_red_lt_50 = red_pixels_lt_50 / channels[2].size

# Display Histogram of a Single Color Channel
plt.figure()
plt.hist(channels[0].ravel(), bins=256, color='red', alpha=0.7)
plt.title('Histogram of Red Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Use Edge Detector on Green Channel
edges = cv2.Canny(green_channel, 100, 200)
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image (Green Channel)')
plt.show()

print("Mean of Green Channel:", mean_green)
print("Fraction of Pixels with Value Smaller than 50 in Red Channel:", fraction_red_lt_50)
