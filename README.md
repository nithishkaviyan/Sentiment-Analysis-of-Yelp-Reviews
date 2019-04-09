# Sentiment-Analysis-of-Yelp-Reviews

This repository contains the code for sentiment analysis performed for Yelp Reviews using <b>Bag-of-Words model</b> with Fully Connected Neural Network, <b>Naive Bayes Classifier</b>, <b> Convolutional Neural Network (from the paper [1])</b> and <b> LSTM Recurrent Neural Network </b>. I am currently working on implementing a Very Deep Convolutional Neural Network from the paper [2].

I have also attached few scripts and notebooks used for preprocessing steps.

The dataset consists of over 6 million reviews given by over 1.5 million users. The pre-processing steps and the selection of subset of reviews are also shown in the attached files.

Test Accuracy for Naive Bayes Classifier: <b>86.37%</b>

Test Accuracy for Bag-of-Words Model    : <b>91.7%</b>

Test Accuracy for Convolutional Neural Network Model : <b> 88.83% </b>

Test Accuracy for LSTM Model    : <b>94.53%</b> (Test Sequence Length = 500)


[1] https://arxiv.org/pdf/1408.5882.pdf

[2] https://arxiv.org/abs/1606.01781
