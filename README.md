# Alphabet Soup Charity Funding Predictor

## Project Overview

This project aims to develop a tool for the nonprofit foundation Alphabet Soup to help select the applicants for funding with the best chance of success. Utilizing a dataset of over 34,000 organizations that have received funding, I aim to create a binary classifier that can predict the effectiveness of the funded projects.

## Table of Contents

1. [Repository Guide](#repository-guide)
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
6. [References](#references)

## Repository Guide

This repository is structured to facilitate easy navigation and understanding of the Alphabet Soup Charity Funding Predictor project. It contains the following main components:

- **Initial Neural Network Folder**: This folder houses the Jupyter notebook used for data preprocessing, a subfolder for model checkpoints saved every 5 epochs during training, and the `AlphabetSoupCharity.h5` file, which contains the saved initial model.

- **Optimized Neural Network Folder**: This folder includes the Jupyter notebook detailing three optimization attempts of the neural network model. It also contains three subfolders, each holding the model checkpoints for one of the three optimized models. The `AlphabetSoupCharity_Optimization.h5` file within this folder stores the "best-performing" optimized model.

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
- Trained for 100 epochs, resulting in a loss of 0.5632 and an accuracy of 72.41%.

#### Results

The initial model demonstrated an accuracy of approximately 72.49%, indicating potential for further optimization to meet the target accuracy of over 75%.

### Model 2: Optimized Neural Network I

#### Model Architecture and Training

The architecture for this optimized model was designed with a strategy focused on enhancing the model's ability to capture complex patterns by significantly increasing the number of neurons in the hidden layers. The rationale behind the choices is as follows:

- Input features: 43
- First hidden layer: 256 neurons, ReLU activation. The number of neurons was chosen by taking the number of input features (43), approximating a multiplier to scale this number up to the next power of 2 that is at least three times larger, resulting in 256.
- Second hidden layer: 128 neurons, ReLU activation. Following the power of 2 rule, this layer has half the number of neurons as the first hidden layer, ensuring a gradual reduction in complexity.
- Output layer: 1 neuron, Sigmoid activation.
- The model was compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric.
- Trained for 100 epochs with a custom callback to save the model's weights every 5 epochs.

#### Results

After training, the model achieved a loss of 0.5764 and an accuracy of 72.34% on the test data. Despite the strategic increase in the network's complexity, the performance did not improve as expected, suggesting that simply increasing the number of neurons might not always result in better outcomes.

### Model 3: Optimized Neural Network II

#### Model Architecture and Training

Building on the previous model, this iteration introduces an additional layer while adhering to the power of 2 rule for neuron counts:

- Input features: 43
- First hidden layer: 256 neurons, ReLU activation. (Chosen as three times the input features, rounded up to the nearest power of 2)
- Second hidden layer: 128 neurons, ReLU activation. (Following the power of 2 rule)
- Third hidden layer: 64 neurons, ReLU activation. (Continuing the power of 2 reduction for additional depth)
- Output layer: 1 neuron, Sigmoid activation.
- The model was compiled and trained identically to the previous models, with the hope that an additional layer would capture more complex patterns.

#### Results

This model iteration resulted in a slight increase in loss and no significant change in accuracy, suggesting that the addition of more layers, even when following a structured approach to their sizing, requires careful consideration of other factors such as overfitting and the complexity of the data.

### Model 4: Optimized Neural Network III

#### Approach and Model Architecture

For this model, a different approach was taken by first applying PCA to reduce the dimensionality of the input features, and then designing the network layers based on the reduced feature set:

- PCA reduced the input features to 32, retaining approximately 98.08% of the variance.
- First hidden layer: 128 neurons, ReLU activation. Here, the choice of 128 neurons follows the principle of starting with a power of 2 that is close to multiplying the reduced features by 3.
- Second hidden layer: 64 neurons, ReLU activation. Continuing with the power of 2 rule for subsequent layers.
- Third hidden layer: 32 neurons, tanh activation. This layer uses a different activation function to introduce non-linearity and complexity.
- Output layer: 1 neuron, Sigmoid activation.
- The model was trained with the same loss function, optimizer, and callbacks as the previous models.

#### Results

The application of PCA and the subsequent network design did not yield the expected improvements, with a significant decrease in accuracy. This outcome highlights the challenge in balancing model complexity, the risk of information loss through dimensionality reduction, and the need for a nuanced approach to layer and neuron configuration.

## Conclusion

Throughout the course of this project, I explored several iterations of neural network models in an attempt to optimize the prediction accuracy for the effective use of funds by organizations funded by Alphabet Soup. Starting with a relatively simpler initial model and progressively introducing complexity through additional layers and dimensionality reduction techniques like PCA, I aimed to enhance the model's predictive capabilities. Surprisingly, the initial, less complex model demonstrated the best performance among the iterations tested. This finding underscores a crucial aspect of model development: more complexity does not necessarily equate to improved performance, especially when the underlying feature set does not support such complexity.

The attempts to improve the model through added layers and PCA did not yield the desired improvements in performance. Instead, these approaches resulted in either marginal improvements or, in the case of PCA, a significant reduction in model accuracy. This suggests that the initial features, though processed and categorized for simplicity, may have been overly simplified, causing a loss of valuable information critical for making accurate predictions.

### Recommendations for Future Work

Based on the outcomes of the various models tested, a recommendation for future attempts at solving this classification problem would be to revisit the preprocessing steps, particularly the handling of categorical variables. The aggregation of rare categories into a single "Other" category, while useful for simplification, may have inadvertently obscured meaningful patterns within the data. Re-evaluating these categories to see if they can be preserved or processed differently could provide the model with a richer feature set that more accurately captures the nuances of the data.

## References
- (1) Callback function documentation - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback 

