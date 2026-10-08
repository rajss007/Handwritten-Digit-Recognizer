# Handwritten Digit Recognition using CNN Model

It is a  machine learning project for recognizing handwritten digits from **0 to 9**.

The project uses the **MNIST dataset** and a CNN (Convolutional Neural Network) made using Keras and TensorFlow.

The main things done in this project are:

- Train a model using the MNIST dataset.
- Test the model using the MNIST test data.
- Give an image to the model and let it predict which digit is in the image.
- Show the prediction and the confidence of the prediction.
---

## Project Structure

The project layout is:

```text
.
├── image/
│   ├── 0.png
│   ├── 00.png
│   ├── ...
│   ├── 9.png
│   └── 99.png
├── train.py
├── test.py
├── evaluating.py
├── mnist_model.keras
├── requirements.txt
└── README.md
```

### File description

| File / Directory | Purpose |
|---|---|
| `train.py` | Loads and cleans MNIST training data, builds and trains the CNN, and saves the trained model. |
| `test.py` | Evaluates the saved model on the MNIST test set and generates a confusion matrix. |
| `evaluating.py` | Predicts a digit from a user-provided image path and displays the prediction. |
| `mnist_model.keras` | Saved trained Keras model used for evaluation and prediction. |
| `requirements.txt` | Pinned Python dependencies required by the project. |
| `image/` | Contains example images that can be supplied to `evaluating.py`. |
| `README.md` | Project documentation and usage instructions. |

---

## Model Architecture

The model is a CNN designed for 28 × 28 grayscale MNIST images.

The architecture includes:

1. Input layer for `(28, 28, 1)`.
2. Rescaling of pixel values by `1/255`.
3. Initial convolution and batch normalization.
4. ReLU activation.
5. Blocks containing:
   - ReLU activation
   - Separable convolution
   - Batch normalization
   - Max pooling
   - Residual projection and addition
6. A final separable convolution block.
7. Global average pooling.
8. Dropout.
9. A dense output layer with 10 classes.

The training and test pipelines remove:
- Images containing non-finite values.
- Completely blank images.

For training, random rotation is used as data augmentation. This helps expose the model to small variations in the orientation of handwritten digits.

---

## Installation
Open the project folder in a terminal.

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Then install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Steps to run the project

There are three main Python files:

```text
train.py
test.py
evaluating.py
```

### 1. Training the Model

To train the model, run:

```bash
python train.py
```

After training, you should have:

```text
mnist_model.keras
```

in the project folder.

---

### 2. Testing the Model

After training, you can test the model using:

```bash
python test.py
```

This uses the MNIST test dataset.

It calculates the test accuracy and also creates a confusion matrix.

---

### 3. Testing Your Own Image

You can also test the model with an individual image.

Run:

```bash
python evaluating.py
```

The program will ask:

```text
Enter image file path:
```

There is an `image` folder in the project, you can use the images inside it.

For example:

```text
image\0.png
```

You can also give the **full location of any other image** that you want to use for testing.

For example:

```text
C:\Users\Pictures\my_digit.png
```

It will then show something like:

```text
Predicted digit: 7
Confidence: 98.52 %
```

It also displays the image with the prediction.

---

## Troubleshooting

### `FileNotFoundError` for `mnist_model.keras`

Make sure the model file exists in the current working directory:

```text
mnist_model.keras
```

If it does not exist, train the model first:

```bash
python train.py
```

### Image cannot be loaded

Check that the path supplied to `evaluating.py` is correct.

For an image in the project folder, use:

```text
image\filename.png
```

Alternatively, provide the complete path to the image.

### Dependency installation problems

Make sure you are using a compatible Python environment and install the exact dependencies with:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended to avoid conflicts with packages installed globally.

---

## Credits / Authors

**Author:** rajss007(Raj Shekhar Sinha)

This project uses the MNIST handwritten digit dataset and libraries such as TensorFlow, Keras, NumPy, and Matplotlib.

---