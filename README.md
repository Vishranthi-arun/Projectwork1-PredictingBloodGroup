## PREDICTING BLOOD GROUP USING FINGERPRINT USING DEEP LEARNING

Blood group prediction is an important task in medical diagnostics, especially for safe blood transfusion, organ transplantation, and managing medical emergencies. 
Traditionally, blood group determination is done manually using blood typing techniques, which involve mixing a blood sample with different antibodies to identify the presence or absence of antigens. 
Although reliable, these methods can be time-consuming, require trained personnel, and may be prone to human error. 
Thus, an automated approach using deep learning has emerged as a promising solution to streamline and enhance blood group classification.
Deep learning techniques, especially convolutional neural networks (CNNs), have demonstrated significant success in image classification tasks. 
In this project, we leverage transfer learning using DenseNet-121, a deep convolutional neural network that has been pre-trained on the ImageNet dataset. 
Transfer learning enables the use of a pre-trained model as a feature extractor, which can be fine-tuned to perform specific tasks with a smaller dataset. 
This method is especially useful when dealing with medical images, where large labeled datasets may not always be available.

## Features
- Utilizes CNNs for accurate blood group classification from fingerprint images.
- Leverages DenseNet121 for powerful feature extraction and high accuracy.
- Trains on fingerprint datasets to identify distinct patterns linked to blood types.
- Supports non-invasive blood group screening, potentially aiding medical diagnostics.
- Enables easy deployment and scalability for real-world healthcare applications.

## Requirements

* Operating System: Requires a 64-bit OS (Windows 10, macOS, or Ubuntu) compatible with deep learning libraries.
* Development Environment: Python 3.7 or later is necessary for scripting and implementing the classification model.
* Deep Learning Framework: TensorFlow (version 2.x) for building, training, and deploying the DenseNet121-based model.
* Image Processing Libraries: Keras and OpenCV for image preprocessing, resizing, and real-time input handling.
* Hardware Requirements: A GPU is recommended to accelerate training and inference.
* IDE: Use Jupyter Notebook or VSCode for interactive coding, model tuning, and visualization.
* Additional Dependencies: Includes scikit-learn for performance metrics, Matplotlib for visualization, and Pandas for data handling and analysis.

## DenseNet-121 Architecture

![DenseNet-121-Architecture](https://github.com/user-attachments/assets/e5c0cb90-c3ce-4c74-adf3-3b22411461bb)

## Flow Chart
![Screenshot 2024-11-12 183107](https://github.com/user-attachments/assets/2d14401c-9ea4-4b9d-b8b7-b5f4946dd38f)

## Program
``
# Step 1: Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Step 2: Define the path to your dataset
dataset_path = "PROJECT/dataset/"  # Local path to dataset

# Step 3: Set important parameters
img_size = (224, 224)  # DenseNet121 expects 224x224 images
batch_size = 32

# Step 4: Preprocessing using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Step 5: Load train and test datasets
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\DELL\PROJECT\dataset\TRAIN',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\DELL\PROJECT\dataset\test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 6: Load the DenseNet121 model with pretrained weights
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 7: Build the model by adding layers on top of DenseNet121
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Adjust according to the number of classes
])
# Freeze the base model layers to retain pre-trained features
base_model.trainable = False

# Step 8: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
# Step 9: Train the model
epochs = 10  # You can adjust the number of epochs
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)

# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 11: Save the trained model locally
model.save('./fingerprint_bloodgroup_model_densenet.h5')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Initialize lists for true labels and predictions
true_labels = []
predicted_labels = []

# Loop through the test generator to obtain predictions and true labels
for images, labels in test_generator:
    # Predict using the model
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)  # Convert probabilities to class indices
    true = np.argmax(labels, axis=1)  # Convert one-hot encoded labels to class indices

    # Extend lists with batch results
    predicted_labels.extend(preds)
    true_labels.extend(true)

    # Stop if we’ve processed the entire test set
    if len(true_labels) >= test_generator.samples:
        break
# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
class_labels = list(test_generator.class_indices.keys())  # Get class names from the generator
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()
# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
def predict_blood_group(model, img_path, class_indices):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Load image and resize to model input size
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted index to the class label
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse the dictionary to map indices to class names
    predicted_class = class_labels[predicted_class_index]

    # Display the image with prediction
    plt.imshow(img)
    plt.title(f"Predicted Blood Group: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# Assuming train_generator is already created and the model is trained
class_indices = train_generator.class_indices  # Get the class indices from the training data generator

# Provide the path to the image you want to test
img_path = r"C:\Users\DELL\PROJECT\DATASET\EX1.jpeg"  

# Predict and display the blood group
predicted_blood_group = predict_blood_group(model, img_path, class_indices)
print("Predicted Blood Group:", predicted_blood_group)
``
## Output
![Screenshot 2024-11-12 183333](https://github.com/user-attachments/assets/c7c2dee6-47cf-49ee-b2fc-563b74a754ab)


Detection Accuracy: 86%
Note: These metrics can be customized based on your actual performance evaluations.

## Model Accuracy
![Screenshot 2024-11-12 185235](https://github.com/user-attachments/assets/1e9a0bed-93f4-491c-8b73-e544f59fbca2)
![Screenshot 2024-11-12 185413](https://github.com/user-attachments/assets/c1e4e3dc-71e2-4492-9856-639820c245d2)


## Result
The blood group classification model, utilizing DenseNet121 as a Convolutional Neural Network (CNN), demonstrates strong performance on both the training and testing datasets:
These results suggest that the blood group classification model is both highly accurate and well-balanced. 
The model successfully predicts blood group types (A, B, AB, O) from fingerprint images, achieving minimal misclassification.
Additionally, the model provides the precise blood group classification based on input fingerprint images, offering real-time predictions and valuable insights for potential integration into healthcare systems.

## Articles published / References
1.  Yue-fang Dong, Wei-wei Fu, Zhe Zhou, Nian Chen, Min Liu and Shi Chen, "ABO blood group detection based on image processing technology," 2017 2nd International Conference on Image, Vision and Computing (ICIVC), Chengdu, 2017, pp. 655-659, doi: 10.1109/ICIVC.2017.7984637.
2.  T. Gupta, "Artificial Intelligence and Image Processing Techniques for Blood Group Prediction," 2024 IEEE International Conference on Computing, Power and Communication Technologies (IC2PCT), Greater Noida, India, 2024, pp. 1022-1028, doi: 10.1109/IC2PCT60090.2024.10486628.  




