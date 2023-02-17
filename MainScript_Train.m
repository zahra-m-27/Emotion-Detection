clc; clear; close all; clc;
%This file calls scripts that form the training pipeline. The training
%faces from pre-sorted and pre-labeled emotions are detected and cropped.
%Then PCA is performed to reduce dimensionality to allow Fisher LDA to be
%performed. Afterwards, eye and mouth regions are detected using Haar-like
%features, or Harris keypoints if Haar fails, and are then extracted. HOG
%features are extracted from these regions, and an SVM is trained.

%This program uses a dataset of front-facing face images to train emotion facial
%expression classifiers and classifies test images.

%Assumes 7 basic emotion categorization: Anger, Contempt,
%Disgust, Fear, Happy, Sad, Surprise

%Crop pre-sorted (by emotion) training images 
CropFaces

%Perform PCA on all the face images to reduce dimensionality
PCA

%Reduce cropped face images to lower dimensionality, then perform
%Fisher LDA. Find thresholds for binary classification 
Fisher

%Perform HOG extraction on training images and train Multi-class SVM
HOG_SVM
