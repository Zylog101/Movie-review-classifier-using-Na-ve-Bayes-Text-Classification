# Movie-review-classifier-using-Na-ve-Bayes-Text-Classification
Developed movie review classifier using Naïve Bayes Text Classification to categorize movie reviews as positive or negative

**Overview**
* Develop a Naïve Bayes model
* From training data with preclassified reviews train the model
* Categorize new reviews into positive and negative category using the model developed
* Utilize thresholding to demonstrate the effect of the same on precision and recall for the test data
* Avoiding underflow by utilizing LogSumExp

**Current Implementation**
* Input
  * Prelabelled (positive and negative) movie reviews
  * Test data composing movie reviews yet to be classified
* Training the model
* Classifying the test data
* Prediction of probabilities for the respective document
* Precision and Recall graph explaining the effect of thresholding on classification

**Future Enhancements**
* Improving the implementation to reduce the time taken for completion
* Improving the accuracy of prediction
* Improving Precision and Recall which currently not produced effectively

**Problem faced during implementation**
* Time required to train was very high initially
  * Solved by reducing the usage of costly splicing operations on csr matrix
* Time required to predict the labels for the entire test data was very high
  * Improved by utilizing dictionary to represent word frequency and utilizing that directly in probability calculation
