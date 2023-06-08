# importing the necessary libraries
import cv2
import pandas
import random
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt


# function to load the train images from the csv file
def load_train_images():
    csv_file = pandas.read_csv('fashion-mnist_train.csv')

    images = csv_file.values
    labels = images.T[0]
    images = np.delete(images, 0, 1)
    images = images.reshape(60000, 28, 28)
    # images = images[0:6000]

    return (images, labels)


# function to load the test images from the csv file
def load_test_images():
    csv_file = pandas.read_csv('fashion-mnist_test.csv')

    images = csv_file.values
    labels = images.T[0]
    images = np.delete(images, 0, 1)
    images = images.reshape(10000, 28, 28)
    # images = images[0:100]

    return (images, labels)


# function to extract the sift features and the labels for the dataset of images
def sift_features(images, labels):
    extractor = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    names = []
    i = -1

    for img in images:
        i += 1
        i %= 10

        img_keypoints, img_descriptors = extractor.detectAndCompute(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), None)

        # if there is no descriptor in any image then remove it from the list
        if(img_descriptors is None):
            continue

        # giving the labels based on the value that was present as label in csv file
        if(labels[i] == 0):
            names.append('T-shirt/top')
        elif(labels[i] == 1):
            names.append('Trouser')
        elif(labels[i] == 2):
            names.append('Pullover')
        elif(labels[i] == 3):
            names.append('Dress')
        elif(labels[i] == 4):
            names.append('Coat')
        elif(labels[i] == 5):
            names.append('Sandal')
        elif(labels[i] == 6):
            names.append('Shirt')
        elif(labels[i] == 7):
            names.append('Sneaker')
        elif(labels[i] == 8):
            names.append('Bag')
        elif(labels[i] == 9):
            names.append('Ankle boot')

        descriptors.append(img_descriptors)

    # list to store all the descriptors together from all the images
    all_descriptors = []
    for img_descriptors in descriptors:
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)

    return (descriptors, all_descriptors, names)


# function to recalculate the clusters for the k mean clustering
def recalculate_clusters(data_set, centroids, k):
    clusters = {}

    for i in range(k):
        clusters[i] = []

    for data in data_set:
        euclid_dist = []
        for j in range(k):
            euclid_dist.append(np.linalg.norm(data - centroids[j]))

        # adding the data to the closest mean
        clusters[euclid_dist.index(min(euclid_dist))].append(data)
    return clusters


# function to recalculate the centroids positions based on the cluster values
def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)

    return centroids


# function to cluster the data given to it and return k centroids of those k clusters
def k_means_clustering(data_set, k=3, repeats=10, centroids={}):
    for i in range(k):
        centroids[i] = data_set[random.randint(0, len(data_set)-1)]

    # recalulating the clusters and the centroids for repeat number of times
    for i in range(repeats):
        clusters = recalculate_clusters(data_set, centroids, k)
        centroids = recalculate_centroids(centroids, clusters, k)

    # converting the dictionary form of centroids to np.array
    centroids1 = []

    for i in range(k):
        centroids1.append(centroids[i])
    centroids = np.array(centroids1)

    return centroids


# function to compute the histograms for image dataset based on the k centroids generated
def ComputeHistogram(img_descriptors, k_mean_centroids):
    img_frequency_vectors = []

    for img_descriptor in img_descriptors:
        img_visual_words = []

        for img_feature in img_descriptor:

            # comparing the distance of each feature with all other features of centroids and storing the lcosest centroid
            dist = MatchHistogram(img_feature, k_mean_centroids[0])
            ind = 0
            for l in range(1, len(k_mean_centroids)):
                dist1 = MatchHistogram(img_feature, k_mean_centroids[l])
                dist = min(dist, dist1)
                if(dist == dist1):
                    ind = l
            img_visual_words.append(ind)
        img_frequency_vector = np.zeros(len(k_mean_centroids))

        # making the histograms for each image sift features based on the centroids features
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        img_frequency_vectors.append(img_frequency_vector)

    return img_frequency_vectors


# function to find the distance between two histograms using euclidean distance method
def MatchHistogram(histogram1, histogram2):
    return np.sum(np.square(histogram1-histogram2))/(np.linalg.norm(histogram1)*np.linalg.norm(histogram2))


# function to find the most closest histogram for test image in the train images
def compare_histograms(test_histogram, train_histograms):
    labels = []

    # # taking the 5 closest train images to the test image
    # for i in range(len(train_histograms)):
    #     dist = MatchHistogram(test_histogram, train_histograms[i])
    #     labels.append([dist, i])

    # labels.sort()

    # return labels[:5]

    # taking the closest train image to the test image
    for i in range(len(train_histograms)):
        dist = MatchHistogram(test_histogram, train_histograms[i])
        labels.append([dist, i])

    labels.sort()

    return labels[0]


# function to get the labels for the test images based on the histograms of the train images
def match_images(train_histograms, test_histograms, train_labels, test_labels):
    pred_labels = []

    # ind = 0

    for test_histogram in test_histograms:
        labels = compare_histograms(test_histogram, train_histograms)
        pred_labels.append(train_labels[labels[1]])

        # i = 0
        # while(i < 5):
        #     if(test_labels[ind] == train_labels[labels[i][1]]):
        #         pred_labels.append(test_labels[ind])
        #         break
        #     i += 1
        # if(i == 5):
        #     pred_labels.append(train_labels[labels[0][1]])
        # ind += 1

    # printing the classification report for the true labels of test images and predicted labels
    print(sklearn.metrics.classification_report(test_labels, pred_labels))


# function to create the visual dictionary to store the closest label of the image to the centroid
def CreateVisualDictionary(k_mean_centroids, train_sift_features, train_labels):
    names = {}

    # finding the closest image to the corresponding centroid and giving its label to the centroid
    for i in range(len(k_mean_centroids)):
        ind = 0
        dist = MatchHistogram(k_mean_centroids[i], train_sift_features[0][0])
        for j in range(len(train_sift_features)):
            for feature in train_sift_features[j]:
                dist1 = MatchHistogram(k_mean_centroids[i], feature)
                dist = min(dist, dist1)
                if(dist == dist1):
                    ind = j
        names[i] = (train_labels[ind])

    # writing the labels to the file named "dictionary.txt"
    with open("dictionary.txt", 'w') as f:
        for key, value in names.items():
            f.write('%s:%s\n' % (key, value))


def execute():
    # taking the train data and its labels by calling the function
    train_data_set, temp_train_labels = load_train_images()

    # extracting the sift feature from the images of train data set
    train_sift_features, all_train_sift_features, train_labels = sift_features(
        train_data_set, temp_train_labels)

    # number of clusters to be formed
    k = 100

    # number of iterations to be performed while finding k means centroids
    itr = 5

    # finding the k means centroids
    k_mean_centroids = k_means_clustering(
        all_train_sift_features, k, itr)

    # getting the histograms for all the train images
    train_histograms = ComputeHistogram(train_sift_features, k_mean_centroids)

    # calling the CreateVisualDictionary function to save the most closest word to the mean of the clusters
    CreateVisualDictionary(k_mean_centroids, train_sift_features, train_labels)

    # taking the test data and its labels by calling the function
    test_data_set, temp_test_labels = load_test_images()

    # extracting the sift feature from the images of test data set
    test_sift_features, all_test_sift_features, test_labels = sift_features(
        test_data_set, temp_test_labels)

    # getting the histograms for all the test images
    test_histograms = ComputeHistogram(test_sift_features, k_mean_centroids)

    # calling the functions to get the labels for all the test images from the train images
    match_images(train_histograms, test_histograms, train_labels, test_labels)


execute()
