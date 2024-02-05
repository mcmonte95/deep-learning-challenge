# Alphabet Soup Charity Funding Predictor

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Model 1: Initial Neural Network](#model-1-initial-neural-network)
   - [Model Architecture and Training](#model-architecture-and-training)
   - [Results](#results)
4. [Model 2: Optimized Neural Network I](#model-2-optimized-neural-network-i)
   - [Model Architecture and Training](#model-architecture-and-training-1)
   - [Results](#results-1)
5. [Model 3: Optimized Neural Network II](#model-3-optimized-neural-network-ii)
   - [Model Architecture and Training](#model-architecture-and-training-2)
   - [Results](#results-2)
6. [Conclusion](#conclusion)
7. [Recommendations for Future Work](#recommendations-for-future-work)

## Project Overview

This project aims to develop a tool for the nonprofit foundation Alphabet Soup to help select the applicants for funding with the best chance of success. Utilizing a dataset of over 34,000 organizations that have received funding, we aim to create a binary classifier that can predict the effectiveness of the funded projects.

## Data Preprocessing

The dataset includes metadata about each organization, such as application type, affiliation, classification, and the amount of funding requested. The preprocessing steps involved:
- Dropping unnecessary identification columns.
- Determining unique values and binning rare categorical variables.
- Encoding categorical variables using `pd.get_dummies()`.
- Splitting the data into feature and target arrays, followed by train-test splitting.
- Scaling the feature datasets.

## Model 1: Initial Neural Network

### Model Architecture and Training

- Input features: 43
- First hidden layer: 80 neurons, ReLU activation.
- Second hidden layer: 30 neurons, ReLU activation.
- Output layer: 1 neuron, Sigmoid activation.
- Compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric.
- Implemented a custom callback to save model weights every 5 epochs.
- Trained for 100 epochs, resulting in a loss of 0.5619 and an accuracy of 72.49%.

### Results

The initial model demonstrated an accuracy of approximately 72.49%, indicating potential for further optimization to meet the target accuracy of over 75%.

## Model 2: Optimized Neural Network I

### Model Architecture and Training

- (Include details from the code provided earlier as this model's architecture and training process)

### Results

(TODO: Add results once provided)

## Model 3: Optimized Neural Network II

### Model Architecture and Training

(TODO: Add details once provided)

### Results

(TODO: Add results once provided)

## Model 4: Optimized Neural Network III

### Model Architecture and Training

(TODO: Add details once provided)

### Results

(TODO: Add results once provided)

## Conclusion

(TODO: Summarize findings and overall project outcomes)

## Recommendations for Future Work

(TODO: Provide recommendations for improving the model or exploring new models)



## References
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
