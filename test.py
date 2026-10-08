import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
_, (x_test, y_test) = mnist.load_data()  #Loading testing data

#Filtering out corrupted images and completely blank images
valid_test = np.isfinite(x_test).all(axis=(1, 2))
not_blank_test = (x_test.sum(axis=(1, 2)) > 0)
test_mask = valid_test & not_blank_test

#Using filtered dataset
x_test_clean = x_test[test_mask]
y_test_clean = y_test[test_mask]

print("Testing set - Original:", len(x_test), ", Cleaned:", len(x_test_clean))

model = keras.models.load_model("mnist_model.keras")

x_test_expanded = np.expand_dims(x_test_clean, axis=-1)

#Predict all test images
predictions = model.predict(x_test_expanded, verbose=1)
scores = keras.ops.softmax(predictions)
predicted_digits = np.argmax(scores, axis=1)

#Calculate test accuracy
test_accuracy = np.mean(predicted_digits == y_test_clean)

print("\nTesting complete!")
print("Test Accuracy: " + str(round(test_accuracy * 100, 2)) +"%")

#Creating confusion matrix
num_classes = 10

#Rows    = True labels
#Columns = Predicted labels
confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

#Filling the confusion matrix
for true_label, predicted_label in zip(y_test_clean,predicted_digits):
    confusion_matrix[true_label, predicted_label] += 1

plt.figure(figsize=(8, 8))

plt.imshow(
    confusion_matrix,
    cmap="Blues"
)

plt.title(
    "MNIST Confusion Matrix\n"
    "Test Accuracy: "+ str(round(test_accuracy * 100, 2))+"%"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar()

plt.xticks(np.arange(10))
plt.yticks(np.arange(10))

plt.gca().invert_yaxis()

#To display the actual values inside each cell
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if confusion_matrix[i, j]
                > confusion_matrix.max() / 2
                else "black"
            )
        )

plt.tight_layout()
plt.show()