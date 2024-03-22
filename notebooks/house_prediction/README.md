# House Price Prediction Using Keras

Exploring how to use Keras package to build a neural network to predict if house prices are above or below median value.

## Prerequisite

### Installation

This project requires Python 3.8 or higher installed in your local machine.
If you have not yet installed any version of Python, you can follow this tutorial to download Anaconda and have Python installed in your local machine.
[Installing Anaconda](https://medium.com/intuitive-deep-learning/getting-started-with-python-for-deep-learning-and-data-science-9ca785560bbc)

If you already have Python installed, you can check the version by running the following command in your terminal.

```bash
python --version
```

### Virtual Environment

#### Python Virtual Environment

To create a virtual environment in python, you can run the following command in your terminal.

```bash
python -m venv house_prediction_venv
```

To activate the virtual environment, you can run the following command in your terminal.

```bash
source house_prediction_venv/bin/activate
```

#### Conda Virtual Environment

To create a virtual environment in conda, you can run the following command in your terminal.

```bash
conda create --name house_prediction_venv
```

To activate the virtual environment, you can run the following command in your terminal.

```bash
conda activate house_prediction_venv
```

### Dependencies

After activating the virtual environment, to run the notebook, you will need to have the dependencies in requirements.txt installed.

```bash
pip install -r requirements.txt
```

In conda environment, you'll need to install pip first by running the following command in your terminal.

```bash
conda install pip
```

### Resources

The dataset used is adapted from Zillow’s Home Value Prediction Kaggle competition data. You can download the dataset from the following link.

[Zillow’s Home Value Prediction Kaggle competition data](https://www.kaggle.com/c/zillow-prize-1/data)

We've reduced the number of input features and changed the task into predicting whether the house price is above or below median value, visit the link below to download the modified dataset and place it in the same directory as your notebook.

[Download dataset](https://drive.usercontent.google.com/uc?id=1GfvKA0qznNVknghV4botnNxyH-KvODOC&export=download)

## Usage

### Data Processing

Before we code any ML algorithm, the first thing we need to do is to put our data in a format that the algorithm will want. In particular, we need to:

1. Read in the CSV (comma separated values) file and convert them to arrays. Arrays are a data format that our algorithm can process.
2. Split our dataset into the input features (which we call x) and the label (which we call y).
3. Scale the data (we call this normalization) so that the input features have similar orders of magnitude.
4. Split our dataset into the training set, the validation set and the test set

### Data Exploration

Exploring our data features by priting the output of the `df` variable:

```python
print(df)
```

We have our input features in the first ten columns:

- Lot Area (in sq ft)
- Overall Quality (scale from 1 to 10)
- Overall Condition (scale from 1 to 10)
- Total Basement Area (in sq ft)
- Number of Full Bathrooms
- Number of Half Bathrooms
- Number of Bedrooms above ground
- Total Number of Rooms above ground
- Number of Fireplaces
- Garage Area (in sq ft)

In our last column, we have the feature that we would like to predict:

- Is the house price above the median or not? (1 for yes and 0 for no)

## Summary

### Neural Network

Coding up this neural network required only a few lines of code:

- We specify the architecture with the Keras Sequential model.
- We specify some of our settings (optimizer, loss function, metrics to track) with `model.compile`
- We train our model (find the best parameters for our architecture) with the training data with `model.fit`
- We evaluate our model on the test set with `model.evaluate`

To deal with overfitting, we can code in the following strategies into our model each with about one line of code:

- L2 Regularization:
  - L2 regularization is a technique used to prevent overfitting in neural
  networks. It adds a penalty term to the loss function, which encourages the model to have smaller weights. This helps to reduce the complexity of the model and prevent it from memorizing the training data too well. By adding L2 regularization, we can improve the generalization ability of the model and reduce the risk of overfitting.
  To add L2 regularization, notice that we’ve added a bit of extra code in each of our dense layers like this:

  ```python
  kernel_regularizer=regularizers.l2(0.01)
  ```

- Dropout
  - This is a technique used in neural networks to prevent overfitting. It randomly sets a fraction of input units to 0 at each update during training, which helps to reduce the reliance on specific input features and encourages the network to learn more robust and generalizable representations.
  To add Dropout, we added a new layer like this:

  ```python
  Dropout(0.3),
  ```

  This means that the neurons in the previous layer has a probability of 0.3 in dropping out during training.

### Consolidated Summary

We’ve written Python code to:

- Explore and Process the Data
- Build and Train our Neural Network
- Visualize Loss and Accuracy
- Add Regularization to our Neural Network

## Extras

Some additional resources for assistance
- [Keras Documentation](https://keras.io/api/)

- [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf)

- [Scikit-learn Documentation](https://scikit-learn.org/0.21/documentation.html)

- [Pandas Documentation](https://pandas.pydata.org/docs/user_guide/index.html)