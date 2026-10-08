import numpy as np
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("mnist_model.keras")
#Getting image filename from the user
image_path = input("Enter image file path: ")

image = keras.utils.load_img(
    image_path,
    color_mode="grayscale",
    target_size=(28, 28)
)

image_array = keras.utils.img_to_array(image)
img_array = np.expand_dims(image_array, axis=0)
predictions = model.predict(img_array, verbose=0)
scores = keras.ops.softmax(predictions[0])
predicted_digit = int(np.argmax(scores))
confidence = float(scores[predicted_digit]) * 100

print("\nPredicted digit:", predicted_digit)
print("Confidence:", round(confidence, 2), "%")

plt.imshow(image_array.squeeze(), cmap="gray")
plt.title("Predicted Digit: "+ str(predicted_digit)+ "\nConfidence: "+ str(round(confidence, 2))+ "%")
plt.axis("off")
plt.show()