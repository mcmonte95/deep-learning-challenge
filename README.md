# Alphabet Soup Charity Funding Predictor

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Initial Model Design](#initial-model-design)
   - [Model 1: Initial Neural Network](#model-1-initial-neural-network)
     - [Model Architecture and Training](#model-architecture-and-training)
     - [Results](#results)
4. [Model Optimization Attempts](#model-optimization-attempts)
   - [Model 2: Optimized Neural Network I](#model-2-optimized-neural-network-i)
     - [Model Architecture and Training](#model-architecture-and-training-1)
     - [Results](#results-1)
   - [Model 3: Optimized Neural Network II](#model-3-optimized-neural-network-ii)
     - [Model Architecture and Training](#model-architecture-and-training-2)
     - [Results](#results-2)
   - [Model 4: Optimized Neural Network III](#model-4-optimized-neural-network-iii)
     - [Approach and Model Architecture](#approach-and-model-architecture)
     - [Results](#results-3)
5. [Conclusion](#conclusion)
6. [Recommendations for Future Work](#recommendations-for-future-work)

## Project Overview

This project aims to develop a tool for the nonprofit foundation Alphabet Soup to help select the applicants for funding with the best chance of success. Utilizing a dataset of over 34,000 organizations that have received funding, we aim to create a binary classifier that can predict the effectiveness of the funded projects.

## Data Preprocessing

The dataset provided by Alphabet Soup contains various metadata about organizations that have received funding. To prepare the data for the neural network model, several preprocessing steps were undertaken to ensure the dataset was suitable for training and prediction. Here are the specific steps taken:

### Dropping Unnecessary Identification Columns

The first step in preprocessing involved removing columns that were not useful for the model. Specifically, the columns `EIN` (Employer Identification Number) and `NAME` (Name of the organization) were dropped. These columns are identifiers that do not contribute to the predictive capability of the model.

### Determining Unique Values and Binning Rare Categorical Variables

The dataset contains several categorical variables with a wide range of values. To simplify the model and improve its performance, rare categorical variables were binned into a single category named "Other". This was applied to the `APPLICATION_TYPE` and `CLASSIFICATION` columns. This process helps in reducing the dimensionality and focusing the model on more common occurrences.

### Encoding Categorical Variables

After simplifying the categorical variables, the next step was to encode these variables using `pd.get_dummies()`. The encoding was applied to all categorical columns, including `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, and `SPECIAL_CONSIDERATIONS`, turning them into numerical data suitable for the neural network.

### Splitting the Data into Feature and Target Arrays

The target variable for our model is `IS_SUCCESSFUL`, which indicates whether the money given to an organization was used effectively. All other columns after encoding and preprocessing were used as features for the model. Thus, the dataset was split into two arrays: `X` for the features and `y` for the target (`IS_SUCCESSFUL`).

### Train-Test Splitting

The preprocessed data was then split into training and testing datasets to evaluate the model's performance. I used the `train_test_split` function from scikit-learn.

### Scaling the Feature Datasets

Finally, the feature data was scaled using `StandardScaler` from scikit-learn. Scaling ensures that the neural network model does not get biased by the scale of the features. 

## Initial Model Design

### Model 1: Initial Neural Network

#### Model Architecture and Training

- Input features: 43
- First hidden layer: 80 neurons, ReLU activation.
- Second hidden layer: 30 neurons, ReLU activation.
- Output layer: 1 neuron, Sigmoid activation.
- Compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric.
- Implemented a custom callback to save model weights every 5 epochs.
- Trained for 100 epochs, resulting in a loss of 0.5619 and an accuracy of 72.49%.

#### Results

The initial model demonstrated an accuracy of approximately 72.49%, indicating potential for further optimization to meet the target accuracy of over 75%.

## Model Optimization Attempts

### Model 2: Optimized Neural Network I

#### Model Architecture and Training

- Input features: 43
- First hidden layer: 256 neurons, ReLU activation. (Increased from 80 in the initial model)
- Second hidden layer: 128 neurons, ReLU activation. (Increased from 30 in the initial model)
- Output layer: 1 neuron, Sigmoid activation. (Unchanged)
- Compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric. (Unchanged)
- The model was trained for 100 epochs with a custom callback to save the model's weights every 5 epochs. (Unchanged)

#### Results

After training, the model achieved a loss of 0.5772 and an accuracy of 72.38% on the test data. This represents a slight decrease in performance compared to the initial model, which had a slightly lower loss of 0.5619 and a marginally higher accuracy of 72.49%. The changes in the neural network's architecture, specifically the increase in neurons in the hidden layers, did not lead to the expected improvement in model performance.

### Model 3: Optimized Neural Network II

#### Model Architecture and Training

- Input features: 43
- First hidden layer: 256 neurons, ReLU activation. (Unchanged from Model 2)
- Second hidden layer: 128 neurons, ReLU activation. (Unchanged from Model 2)
- Third hidden layer: 64 neurons, ReLU activation. (An additional layer compared to Model 2)
- Output layer: 1 neuron, Sigmoid activation. (Unchanged)
- The model was compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric. (Unchanged)
- The model was trained for 100 epochs with a custom callback to save the model's weights every 5 epochs. (Unchanged)

#### Results

This model iteration resulted in a loss of 0.6028 and an accuracy of 72.49% on the test data. Compared to Model 2, which had a slightly lower loss (0.5772) but the same accuracy (72.38%), adding an additional hidden layer with 64 neurons did not significantly improve performance. The slight increase in loss suggests that simply adding more layers without adjusting other parameters may not yield better results.

### Model 4: Optimized Neural Network III

#### Approach and Model Architecture

- PCA was applied to the scaled feature data, reducing the number of input features to 32 from the original 43, while retaining approximately 98.08% of the variance in the dataset.
- Input features after PCA: 32
- First hidden layer: 128 neurons, ReLU activation.
- Second hidden layer: 64 neurons, ReLU activation.
- Third hidden layer: 32 neurons, tanh activation. (Introduced a different activation function)
- Output layer: 1 neuron, Sigmoid activation.
- The model was compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric.
- The model was trained for 100 epochs with a custom callback to save the model's weights every 5 epochs. (Unchanged)

#### Results

After applying PCA and reconfiguring the neural network, the model achieved a loss of 0.8329 and an accuracy of 52.97% on the test data. This represents a significant decrease in performance compared to the previous models. The results indicate that while PCA can be a powerful tool for dimensionality reduction and can help in simplifying the input data, in this case, it led to a loss of critical information necessary for the neural network to achieve high accuracy.

## Conclusion

(TODO: Summarize findings and overall project outcomes)

## Recommendations for Future Work

(TODO: Provide recommendations for improving the model or exploring new models)



## References
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
