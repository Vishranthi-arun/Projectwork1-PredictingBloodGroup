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




