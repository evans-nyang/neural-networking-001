# House Prediction Using Keras

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

### Data Exploration and Processing

Before we code any ML algorithm, the first thing we need to do is to put our data in a format that the algorithm will want. In particular, we need to:

1. Read in the CSV (comma separated values) file and convert them to arrays. Arrays are a data format that our algorithm can process.
2. Split our dataset into the input features (which we call x) and the label (which we call y).
3. Scale the data (we call this normalization) so that the input features have similar orders of magnitude.
4. Split our dataset into the training set, the validation set and the test set

#### Reading in the CSV file

We can use the pandas library to read in the CSV file. The pandas library is a powerful library that allows us to manipulate data easily.

```python
import pandas as pd

# Read in the CSV file
df = pd.read_csv('house_prediction.csv')
```
