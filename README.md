# Handwritten Digit Recognition with Voice Integration

![Demo](![image](https://github.com/user-attachments/assets/f3df7919-e609-4210-aff5-bb8bfdbfb8a7)
) <!-- Link to your demo GIF or image -->

## Overview
The **Handwritten Digit Recognition** project utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from 0 to 9. This application is enhanced with a voice integration feature using `pyttsx3`, allowing users to interact with the application using voice commands. The user-friendly interface is built with Tkinter, providing a canvas for users to draw digits.

## Features
- **Digit Recognition**: Draw a digit on the canvas and the model predicts which digit it is.
- **Voice Feedback**: The application provides voice feedback of the recognized digit using `pyttsx3`.
- **Clear Canvas**: A button to clear the canvas for new input.
- **User-Friendly Interface**: Built using Tkinter for a seamless user experience.

## How It Works
1. **Drawing on Canvas**: Users can draw a digit on a Tkinter canvas.
2. **Image Preprocessing**: The drawn image is processed to match the input requirements of the pre-trained MNIST model (28x28 grayscale image).
3. **Model Prediction**: The processed image is fed into the CNN model, which outputs the predicted digit.
4. **Voice Output**: The predicted digit is announced using voice synthesis provided by `pyttsx3`.

### Machine Learning Model
The core of this project is a CNN model trained on the MNIST dataset, which contains thousands of handwritten digits. The CNN architecture is capable of learning complex patterns and features from the images, allowing it to accurately recognize digits based on the pixel data.

The model follows these steps:
- **Training**: The model is trained on labeled data (images of digits) to minimize prediction error.
- **Evaluation**: The model is evaluated on a separate test set to ensure it generalizes well to unseen data.
- **Prediction**: When a user draws a digit, the model preprocesses the image and predicts the corresponding digit.

## Technologies Used
- **Python**: The programming language used for this application.
- **Tkinter**: For creating the graphical user interface (GUI).
- **pyttsx3**: A text-to-speech conversion library that provides voice feedback.
- **TensorFlow/Keras**: For building and training the Convolutional Neural Network model.
- **NumPy**: For numerical operations and image preprocessing.

## Installation

### Prerequisites
- Python 3.x
- Required libraries: TensorFlow, Keras, NumPy, Tkinter, pyttsx3

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
