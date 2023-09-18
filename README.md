# Machine-Learning with Big-Data on different scenarios

These Notebooks with their Question Statement and Reports, came under the course CSL7110, taken by Prof. Yashasvi Verma.

## Lab-1

![2-Figure1-1](https://github.com/kwanit1142/Machine-Learning-with-Big-Data-on-different-scenarios/assets/54277039/27fc7e7d-a202-48aa-b26e-d9f75f55a67c)

Download the MNIST dataset. Pick its training and test sets which contain 60k samples, each represented by a 784-dimensional vector. There are 10 classes, and each sample is associated with one class. Preprocess the features using the appropriate technique(s) and store them in a file.

a.) Implement the BFR and CURE clustering algorithms on this data, assuming that you can store 'K1' samples in their main memory at a time. One may use any existing libraries for performing the initial k-means clustering in the case of BFR and agglomerative clustering in the case of CURE. After this, every step needs to be implemented from scratch.

b.) For the BFR algorithm, ensure that the number of clusters obtained at the end of the training the process is 10.

c.) For the CURE algorithm, keep the number of clusters as 10.

d.) After clustering, calculate the percentage of samples from each class and convert it into probability values. Using these, calculate the entropy of each cluster. Also, calculate the total entropy of all the clusters by summing the entropy of individual clusters.

e.) Re-run the 2 algorithms 5 times assuming K1 = {{100,200,500} and report the above result.

![qBBEmoQ](https://github.com/kwanit1142/Machine-Learning-with-Big-Data-on-different-scenarios/assets/54277039/2f7aada6-a0b9-43f0-b747-456b927cfb12)

## Lab-2

![1_27nQOTC79yfh5lzmL06Ieg](https://github.com/kwanit1142/Machine-Learning-with-Big-Data-on-different-scenarios/assets/54277039/37b9b9bf-b47b-4a9c-842f-a872fc0220f5)

Download the MNIST dataset. Pick its training and test sets, each represented by a 784-dimensional vector. There are 10 classes, and each sample is associated with one class. In this, we will use the raw binary feature vectors, assuming they are 'shingles'.

a.) Write a code to classify the test samples using the kNN algorithm and Jaccard similarity by varying the value of k in {1,2,3,4,5}. Report the classification accuracy and time required to classify all the test samples using one CPU core.

b.) Using any publicly available code for LSH, classify the test samples using the kNN algorithm. Vary the length of the signature vector in {40,60} and 's' in {0.8,0.9}. For each combination, run the experiment multiple times to calculate the average and standard deviation of the classification accuracy and time required to classify all the test samples in all the set-ups using one CPU core.

c.) Compare a.) and b.) in terms of classification accuracy, the time required, and peak RAM required.

## Lab-3

![1_aFhOj7TdBIZir4keHMgHOw](https://github.com/kwanit1142/Machine-Learning-with-Big-Data-on-different-scenarios/assets/54277039/80bb4048-ad25-43bc-ad7c-c5b43fb8d7c2)

Download the MNIST dataset. Pick its training and test sets, each represented by a 784-dimensional vector. There are 10 classes, and each sample is associated with one class.

a.) Using the raw binary features, implement the streaming Naive Bayes algorithm and classify the test data.

b.) Project the binary features into a lower dimensional space using a dimensionality reduction technique of your choice (such as PCA, LDA, t-SNE, etc.), by varying the dimensionality in the range {50,100,200}, and classify the test data using the same algorithm.

c.) Compare a.) and b.) in terms of classification accuracy and time required during the training and testing phase.

d.) Compare a.) and b.) with the classification accuracy you had obtained in assignment-2.
