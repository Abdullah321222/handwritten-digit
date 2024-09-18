import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import speech_recognition as sr
import threading

engine = pyttsx3.init()
recognizer = sr.Recognizer()
model = load_model('mnist_model.h5')  

def speak(text):
    """Speak out a text."""
    engine.say(text)
    engine.runAndWait()
    l1=Label()

def predict_digit(image):
    """Predict the digit from the drawn image."""
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image) / 255.0  
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)  
    return predicted_digit, confidence

def paint(event):
    """Paint on the canvas based on mouse motion."""
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=20)
    draw.line([x1, y1, x2, y2], fill="black", width=20)

def on_predict():
    """Handle the prediction when the user clicks 'Predict' or says 'predict'."""
    global image, draw
    predicted_digit, confidence = predict_digit(image)
    result_text.set(f'Prediction: {predicted_digit} (Confidence: {confidence:.2f})')
    speak(f"I believe the digit you drew is {predicted_digit} with a confidence of {confidence:.2f}")

def reset_canvas():
    """Reset the canvas and image for new drawing."""
    global image, draw
    canvas.delete("all")
    image = Image.new("RGB", (280, 280), "white")
    draw = ImageDraw.Draw(image)
    result_text.set("Prediction: None")
    speak("Canvas cleared, please start drawing the digit.")

def listen_for_command():
    """Listen for a spoken command to predict or clear."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            if "predict" in command:
                on_predict()
            elif "clear" in command:
                reset_canvas()
        except:
            pass  

def start_listening():
    """Start a separate thread for continuous listening."""
    listen_thread = threading.Thread(target=listen_for_command, daemon=True)
    listen_thread.start()


root = tk.Tk()
root.title("Digit Recognizer")
root.configure(bg='light pink')
L1 = Label(root, text="Handwritten Digit Recoginition", font=('Algerian', 12), fg="blue")
L1.place(x=3, y=4)

main_frame = Frame(root, bg='light pink')
main_frame.pack(pady=20)

canvas = tk.Canvas(main_frame, width=560, height=560, bg='white')
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
image = Image.new("RGB", (580, 580), "white")
draw = ImageDraw.Draw(image)
canvas.bind("<B1-Motion>", paint)


result_text = tk.StringVar()
result_label = Label(main_frame, textvariable=result_text, font=('Helvetica', 18), bg='light blue')
result_text.set("Prediction: None")
result_label.grid(row=1, column=0, columnspan=2)


predict_button = Button(main_frame, text="Predict", command=on_predict, font=('Helvetica', 18), bg='light green')
predict_button.grid(row=2, column=0, pady=20, padx=10)

clear_button = Button(main_frame, text="Clear", command=reset_canvas, font=('Helvetica', 18), bg='salmon')
clear_button.grid(row=2, column=1, pady=20, padx=10)

speak("Welcome! Please start drawing the digit. Say 'Predict' to predict or 'Clear' to start over.")
start_listening()
root.mainloop()
