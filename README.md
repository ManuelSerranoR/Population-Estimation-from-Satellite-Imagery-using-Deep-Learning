# Population-Estimation-from-Satellite-Imagery-using-Deep-Learning

This repository contains material related to the project "Population Estimation from Satellite Imagery using Deep Learning" for the course Introduction to Data Science, at New York University, in Fall 2017.

Group members:
- Manuel Serrano
- Santiago Novoa
- Yassine Kadiri
- Zsolt Pajor-Gyulai

We attempted to build models predicting population density based on satellite imagery inspired. We achieve this by comparing spatially disaggregated census data on the continental United States (that is excluding Alaska and U.S. territories) to satellite images of the particular region. We use several approaches:
1. Naive logistic regression on the vectorized satellite images.
2. Convolutional Neural Network (CNN) built from scratch in Tensorflow.
3. Pre-trained CNN developed for image recognition (Vgg16).
