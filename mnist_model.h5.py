import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    

    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
  
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels):
    model.fit(train_images, train_labels, epochs=20, validation_split=0.1)
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'\nTest accuracy: {test_accuracy:.4f}')

    return model

def save_model(model, filename='mnist_model.h5'):
    model.save(filename)
    print(f'Model saved as {filename}')

def load_model_for_prediction(filename='mnist_model.h5'):
    return tf.keras.models.load_model(filename)

def make_prediction(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    model = build_model()
    model = train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels)
    save_model(model)
    

    model = load_model_for_prediction()
    test_image = test_images[0:1]  
    predicted_class = make_prediction(model, test_image)
    actual_class = np.argmax(test_labels[0])
    print(f'Predicted class: {predicted_class}, Actual class: {actual_class}')
