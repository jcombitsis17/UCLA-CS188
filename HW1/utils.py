import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
import random
from scipy.spatial import distance

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []

    for i, label in enumerate(train_classes):
        if label != '.DS_Store':
            for filename in os.listdir(train_path + label + '/'):
                image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
                train_images.append(image)
                train_labels.append(i)  
    for i, label in enumerate(test_classes):
        if label != '.DS_Store':
            for filename in os.listdir(test_path + label + '/'):
                image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
                test_images.append(image)
                test_labels.append(i)
                
    return train_images, test_images, train_labels, test_labels

def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    
    classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='kd_tree', metric='euclidean')
    classifier.fit(train_features, train_labels)

    predicted_categories = classifier.predict(test_features)

    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    num_categories = 15
    classifiers = []
    predicted_categories = []
    for i in range(1, num_categories + 1):
        if (is_linear):
            classifier = svm.SVC(C=svm_lambda, kernel='linear', gamma='scale', class_weight = 'balanced')
        else:
            classifier = svm.SVC(C=svm_lambda, kernel="rbf", gamma='scale', class_weight = 'balanced')
        
        #get labels as 0 or 1, depending on if it is label 'i'
        train_labels = np.array(train_labels)
        labels = train_labels == i
        labels = labels.astype(int)

        classifier.fit(train_features, labels)
        classifiers.append(classifier)

    for feature in test_features:
        decision = [classifier.decision_function([feature])[0] for classifier in classifiers]
        predicted_categories.append(np.array(decision).argmax()+1)

    return predicted_categories

def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.

    resized_image = cv2.resize(input_image, (target_size, target_size))
    output_image = cv2.normalize(resized_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

    return output_image

def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)

    n = len(true_labels)
    intersection = [i for i in range(n) if true_labels[i] == predicted_labels[i]]

    accuracy = len(intersection)/n * 100
    return accuracy

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    
    num_features = 25
    detection = None
    if feature_type == "sift":
        detection = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
    elif feature_type == "surf":
        detection = cv2.xfeatures2d.SURF_create()
    elif feature_type == "orb":
        detection = cv2.ORB_create(nfeatures=num_features)    

    features = []
    for img in train_images:
        _,desc = detection.detectAndCompute(img, None)
        # surf doesnt have nfeatures property so we have to manually select 20 features
        if feature_type == 'surf' and len(desc) > num_features:
            desc = random.sample(list(desc), num_features)
        if (desc is not None):
            for descriptor in desc:
                features.append(descriptor)

    if clustering_type == 'kmeans':
        clusters = cluster.KMeans(n_clusters=dict_size)
        vocabulary = clusters.fit(np.array(features)).cluster_centers_
    elif clustering_type == 'hierarchical':
        clusters = cluster.AgglomerativeClustering(n_clusters=dict_size)
        labels = clusters.fit(np.array(features)).labels_

        vocabulary = []
        features = np.array(features)
        for i in range(dict_size):
            cluster_center = np.mean(features[labels == i], axis=0)
            vocabulary.append(cluster_center)
        vocabulary = np.array(vocabulary)

    return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    detection = None
    if feature_type == "sift":
        detection = cv2.xfeatures2d.SIFT_create()
    elif feature_type == "surf":
        detection = cv2.xfeatures2d.SURF_create()
    elif feature_type == "orb":
        detection = cv2.ORB_create()    

    _, desc = detection.detectAndCompute(image, None)

    n_bins = len(vocabulary) 

    #sometimes orb returns none, so return a histogram of zeros
    if (desc is None):
        return np.zeros(n_bins)
    
    dist = distance.cdist(desc, vocabulary, 'euclidean')
    buckets = np.argmin(dist, axis=1)
    Bow,_ = np.histogram(buckets, n_bins)
    Bow = preprocessing.normalize(Bow[:,np.newaxis],axis=0).ravel()

    return Bow

def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    classResult = []

    for size in [8, 16, 32]:
        for n in [1, 3, 6]:
            start = timeit.default_timer()
            resized_train  = [ imresize(image, size).flatten() for image in train_features ]
            resized_test   = [ imresize(image, size).flatten() for image in test_features  ]

            #model = neighbors.KNeighborsClassifier(n)
            #model.fit(resized_train, train_labels)
            #predicted_labels = model.predict(resized_test)
            predicted_labels = KNN_classifier(resized_train, train_labels, resized_test, n)
            runtime = timeit.default_timer() - start
            
            accuracy = reportAccuracy(test_labels, predicted_labels)
            classResult.append(accuracy)
            classResult.append(runtime)

    return classResult
    