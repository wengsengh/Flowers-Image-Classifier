# Flowers Image Classifier

This project was completed as part of the Udacity Data Analyst Nanodegree program requirements

## Project Overview

This project uses PyTorch, torchvision package and  transfer learning to train pre-trained ImageNet neural networks to identify the [flowers species](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 

The walks through conducted in a Jupyter Notebook for the image classifier and use the example in test image for prediction. Then the classifier converted to executable python scripts with argparse to provide execution options. Which can be run from command line using Python "train.py" and "predict.py".

## What do I need to install?

- Jupyter Notebooks
- PyTorch
- Libraries: numpy, matplotlib.pyplot, json, os, random, collections, PIL, argparse

## Project Details

The 8 Steps for the image classifier:
1. Load Dataset
2. Transform the Dataset
3. Create Model
4. Train Model
5. Save the Model
6. Load the Model
7. Predict the Image
8. Display the result

The final model using initial learning rate of 0.001, Adam optimizer, and trained on 5 epochs with 83% accuracy for the validation set.
