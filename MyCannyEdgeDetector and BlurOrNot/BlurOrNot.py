import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from skimage import feature
import os

# Function to get the threshold value


def dataset():
    path = (r"C:\Users\mitta\Downloads\images\\")

    x = np.zeros(100)
    y = np.zeros(100)
    z = np.zeros(100)

    listing = os.listdir(path)

    i = 0

    for file in listing:
        image = io.imread(path+file)
        x[i] = i
        y[i] = filters.laplace(image).var()
        i += 1

    path = (r"C:\Users\mitta\Downloads\images1\\")

    listing = os.listdir(path)

    i = 0

    for file in listing:
        image = io.imread(path+file)
        z[i] = filters.laplace(image).var()
        i += 1

    # Red are the sharp images, blue are the blur images
    plt.scatter(x, y, color='red')
    plt.scatter(x, z, color='blue')
    plt.yticks(np.arange(0, 0.1, 0.01))
    plt.xlabel('image')
    plt.ylabel('score')
    plt.title('Variance Graph')
    plt.show()


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

# Function to get the Laplacian Kernel convolved with the image

def laplacian_filter(temp_image):
    temp_image = rgb2gray(temp_image)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolution(temp_image, kernel)

# Function to find the probability of an image to be blur or non blur using variance

def probability(variance):
    if(variance > 0.01):
        text = "Not Blur, Probabilty = "+str(round(min(1, variance/0.02), 2))
        plt.title(text)
    else:
        text = "Blur, Probabilty = "+str(round(max(0, 1-variance/0.02), 2))
        plt.title(text)

# Function to check if image is blur or not

def BlurOrNot(image):
    variance = laplacian_filter(image).var()
    plt.imshow(image)
    plt.axis('off')
    probability(variance)
    plt.show()


# dataset()

# path = (r"C:\Users\mitta\Downloads\2020csb1113\images\\")
# listing = os.listdir(path)

# for file in listing:
#     image = io.imread(path+file)
#     BlurOrNot(image)

print("Enter the Image Address : ")
address = input()
image = io.imread(address)
BlurOrNot(image)
