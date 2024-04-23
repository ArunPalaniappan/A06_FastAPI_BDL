# Importing the required libraries
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import numpy as np
import os
from PIL import Image
import io

# Intiating the FastAPI app module
app = FastAPI()

# Function to load the trained model
def load_model(path: str) -> keras.Sequential:
    model = keras.models.load_model(path)
    return model

# Load the model from the path provided as environment arguement
# $env:MODEL_PATH="C:\Users\91979\Desktop\Jup_NoteBks\BDL\Asgt_6\model\mnist_exp_2.h5"
model_path = os.getenv("MODEL_PATH")
final_model = load_model(model_path)

# Function to format the image to match the input size of the model
def format_image(img):
    img_array = np.array(img.resize((28, 28))) # Resize the image to 28x28
    return img_array

# Function to predict the digit from the formatted image data
def predict_digit(model, data_point: list) -> str:
    data = np.array(data_point).reshape(-1, 784) / 255.0 # Normalize the data
    prediction = model.predict(data) # Make predictions
    digit = np.argmax(prediction) # Get the index of the highest prediction
    return str(digit)

# Root(bootup) endpoint
@app.get("/")
def read_root():
    return {"This is an": "MNIST app"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()  # Read the uploaded file
    img = Image.open(io.BytesIO(contents)).convert('L')  # Open the image and convert to grayscale
    img_array = format_image(img)  # Format the image
    data_point = img_array.flatten().tolist()  # Flatten the image array and convert to list
    digit = predict_digit(final_model, data_point)  # Predict the digit
    return {"digit": digit}  # Return the predicted digit as response


