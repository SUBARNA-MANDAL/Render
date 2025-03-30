from PIL import Image
from fastapi import FastAPI,File,UploadFile
import uvicorn
from pydantic import BaseModel
import requests
from fastapi.responses import FileResponse
from pathlib import Path
import numpy as np
import tensorflow as tf
import json


MODEL = tf.keras.models.load_model("MobileNet_rice.h5",compile=False)

app = FastAPI()

@app.get("/")
async def start():
    return "Welcome.."
@app.get("/ping")
async def ping():
    return "Hello, I am alive..."


# Directory to temporarily store the image
UPLOAD_DIR = Path("downloads/")
UPLOAD_DIR.mkdir(exist_ok=True)  # Ensure the directory exists


# Define the structure of the incoming JSON request
class ImageUrl(BaseModel):
    url: str  # The URL of the image to download


@app.post("/image_predict")
async def image_predict(image_url: ImageUrl):
    url = image_url.url

    # Check if the URL is valid by attempting to download the image
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for bad responses
    except requests.exceptions.RequestException as e:
        return {"error": f"Unable to download the image: {str(e)}"}



    file_name = "downloaded_image.jpg"  # Use a default name
    file_location = UPLOAD_DIR / file_name

    # Save the image locally
    try:
        with open(file_location, "wb") as buffer:
            for chunk in response.iter_content(1024):
                buffer.write(chunk)
        # print(f"Image saved at {file_location}")  # For debugging purposes
    except Exception as e:
        return {"error": f"Failed to save the image: {str(e)}"}


    def preprocess_image(image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array



# Preprocess the image
    image_array = preprocess_image("downloads/downloaded_image.jpg")

    # Make prediction
    predictions = MODEL.predict(image_array)
    
    # Assuming a classification model, you can get the class index or label
    predicted_class = np.argmax(predictions, axis=1).item()  # Get the class index (adjust based on your model)


    # Load disease mapping from JSON file
    with open("class_indices.json") as f:
        disease_mapping = json.load(f)

    # Assume the predicted class is obtained from the model (e.g., class 2)
    # predicted_class = 2

    # Get the disease name based on the predicted class
    disease_name = disease_mapping.get(str(predicted_class), "Unknown disease")



    return {"predicted_class": predicted_class, "disease_name": disease_name}

    # Return the prediction
    # return {"predicted_class": predicted_class}


    # Return the image as a response
    # return FileResponse(file_location)









if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
