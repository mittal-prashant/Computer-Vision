import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage import feature
from skimage import metrics
import cv2
import os

# Function to get the gaussian kernel of size (2*length+1 x 2*length+1)


def gaussian_kernel(length=2, sigma=2):
    kernel = np.zeros((2*length+1, 2*length+1))
    val1 = 2*np.pi*(sigma*sigma)

    for x in range(-length, length+1):
        for y in range(-length, length+1):
            # val2 = np.exp(-(x*x + y*y)/(2 * sigma*sigma))
            val2 = np.exp(-(x*x + y*y)/(2*sigma*sigma))
            kernel[x+length, y+length] = val2/val1

    kernel = kernel/kernel.sum()

    return kernel

# Convolution function to convolve an image with a given kernel and return convolved image


def convolution(temp_image, kernel):
    width, height = temp_image.shape
    length = kernel.shape[0]//2
    image = np.zeros([width, height])

    for i in range(0, width):
        for j in range(0, height):
            # Corner and edge pixels are given the value which would be in convolved image at distance length from edge
            sub_image = temp_image[max(length, min(width-length-1, i))-length:max(length, min(width-length-1, i)) +
                                   length+1, max(length, min(height-length-1, j))-length:max(length, min(height-length-1, j))+length+1]
            image[i, j] = np.sum(np.multiply(sub_image, kernel))

    return image

# Function to apply sobel filter process on the image
# Returns the image on which sobel filter is applied and the gradient of every pixel


def sobel_filter(temp_image):
    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image_x = convolution(temp_image, dx)

    dy = dx.transpose()
    image_y = convolution(temp_image, dy)

    image = np.sqrt(np.square(image_x) + np.square(image_y))
    image = image / image.max() * 255
    gradient = np.arctan(image_y, image_x)

    return (image, gradient)

# Function to perform the non_maximum suppression to get thin fine edges by keeping only pixels with greater value


def non_max(temp_image, gradient):
    width, height = temp_image.shape
    image = np.zeros([width, height])

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            direction = gradient[x, y]

            if ((0 <= direction < np.pi / 8) or (15 * np.pi / 8 <= direction <= 2 * np.pi)
                    or (7*np.pi / 8 <= direction < 9 * np.pi / 8)):
                last_pixel = temp_image[x, y - 1]
                next_pixel = temp_image[x, y + 1]

            elif ((np.pi / 8 <= direction < 3 * np.pi / 8) or (9 * np.pi / 8 <= direction < 11 * np.pi / 8)):
                last_pixel = temp_image[x + 1, y - 1]
                next_pixel = temp_image[x - 1, y + 1]

            elif ((3 * np.pi / 8 <= direction < 5 * np.pi / 8) or (11 * np.pi / 8 <= direction < 13 * np.pi / 8)):
                last_pixel = temp_image[x + 1, y]
                next_pixel = temp_image[x - 1, y]

            else:
                last_pixel = temp_image[x + 1, y + 1]
                next_pixel = temp_image[x - 1, y - 1]

            if temp_image[x, y] >= last_pixel and temp_image[x, y] >= next_pixel:
                image[x, y] = temp_image[x, y]

    return image

# This is double threshold function to determine the strong, weak and non-relevant pixels


def double_threshold(temp_image, Low_Threshold=0.04, High_Threshold=0.09):
    highThreshold = temp_image.max() * High_Threshold
    lowThreshold = highThreshold * Low_Threshold

    # print(highThreshold, lowThreshold)

    width, height = temp_image.shape
    image = np.zeros([width, height])

    weak = 25
    strong = 255

    weak_x, weak_y = np.where(
        (temp_image <= highThreshold) & (temp_image >= lowThreshold))
    strong_x, strong_y = np.where(temp_image >= highThreshold)

    image[weak_x, weak_y] = weak
    image[strong_x, strong_y] = strong

    return image

# Hystresis Function to convert weak pixels into strong pixels if it is near a strong pixel


def hysteresis(temp_image):
    width, height = temp_image.shape
    weak = 25
    strong = 255

    for i in range(1, width-1):
        for j in range(1, height-1):
            if (temp_image[i, j] == weak):
                if ((temp_image[i-1, j-1] == strong) or (temp_image[i-1, j] == strong) or (temp_image[i-1, j+1] == strong)
                    or (temp_image[i, j-1] == strong) or (temp_image[i, j+1] == strong)
                        or (temp_image[i+1, j-1] == strong) or (temp_image[i+1, j] == strong) or (temp_image[i+1, j+1] == strong)):
                    temp_image[i, j] = strong
                else:
                    temp_image[i, j] = 0

    return temp_image


# Self Made Canny Edge Detector function which goes over many processes to get the desired result

def myCannyEdgeDetector(image, Low_Threshold=0.04, High_Threshold=0.09):
    image = rgb2gray(image)
    gaussian_image = convolution(image, gaussian_kernel())
    sobel_image, gradient = sobel_filter(gaussian_image)
    non_max_image = non_max(sobel_image, gradient)
    threshold_image = double_threshold(
        non_max_image, Low_Threshold, High_Threshold)
    outputImage = hysteresis(threshold_image)
    outputImage = outputImage/outputImage.max()*255
    inbuilt_canny_image = feature.canny(image)
    inbuilt_canny_image = inbuilt_canny_image/inbuilt_canny_image.max()*255

    print("PSNR :", metrics.peak_signal_noise_ratio(
        outputImage, inbuilt_canny_image, data_range=255), "dB")
    print("SSIM :", metrics.structural_similarity(
        outputImage, inbuilt_canny_image))

    fig, axes = plt.subplots(1, ncols=3, figsize=(24, 16))
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('original image')
    axes[1].imshow(outputImage, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Self Made Canny Edge Detector')
    # axes[2].imshow(feature.canny(image, sigma=2, low_threshold=0.06, high_threshold=0.09), cmap='gray')
    axes[2].imshow(inbuilt_canny_image, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Inbuilt Canny Edge Detector')
    plt.show()

# path = (r"C:\Users\mitta\Downloads\2020csb1113\images\\")
# listing = os.listdir(path)

# for file in listing:
#     image = io.imread(path+file)
#     myCannyEdgeDetector(image, 0.04, 0.09)


print("Enter the Image Address : ")
address = input()
image = io.imread(address)
myCannyEdgeDetector(image, 0.04, 0.09)
