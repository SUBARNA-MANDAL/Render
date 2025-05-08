# import math
# from PIL import Image
# from fastapi import FastAPI,File,UploadFile
# import uvicorn
# from pydantic import BaseModel
# import requests
# from fastapi.responses import FileResponse
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# import json


# MODEL_rice = tf.keras.models.load_model("rice/MobileNet_rice.h5",compile=False)
# MODEL_potato = tf.keras.models.load_model("potato/MobileNet_potato.h5",compile=False)
# MODEL_tomato = tf.keras.models.load_model("tomato/MobileNet_tomato.h5",compile=False)


# app = FastAPI()

# @app.get("/")
# async def start():
#     return "Welcome.."
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive..."


# # Directory to temporarily store the image
# UPLOAD_DIR = Path("downloads/")
# UPLOAD_DIR.mkdir(exist_ok=True)  # Ensure the directory exists


# # Define the structure of the incoming JSON request
# class ImageUrl(BaseModel):
#     imageUrl: str
#     modelId: str


# @app.post("/image_predict")
# async def image_predict(image_url: ImageUrl):
#     url = image_url.imageUrl
#     id = image_url.modelId

#     # Check if the URL is valid by attempting to download the image
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise exception for bad responses
#     except requests.exceptions.RequestException as e:
#         return {"error": f"Unable to download the image: {str(e)}"}



#     file_name = "downloaded_image.jpg"  # Use a default name
#     file_location = UPLOAD_DIR / file_name

#     # Save the image locally
#     try:
#         with open(file_location, "wb") as buffer:
#             for chunk in response.iter_content(1024):
#                 buffer.write(chunk)
#         # print(f"Image saved at {file_location}")
#     except Exception as e:
#         return {"error": f"Failed to save the image: {str(e)}"}


#     def preprocess_image(image_path):
#         image = Image.open(image_path)
#         image = image.resize((224, 224))
#         image_array = np.array(image) / 255.0  # Normalize
#         image_array = np.expand_dims(image_array, axis=0)  # batch dimension
#         return image_array



# # Preprocess the image
#     image_array = preprocess_image("downloads/downloaded_image.jpg")

#     # Make prediction


#     if id=="1" :
#         predictions = MODEL_rice.predict(image_array)
#         #get the class index or label
#         predicted_class = np.argmax(predictions, axis=1).item()  # Get the class index
#         # Load disease mapping from JSON file
#         with open("rice/class_indices.json") as f:
#             disease_mapping = json.load(f)
#     elif id=="2" :
#         predictions = MODEL_potato.predict(image_array)
#         predicted_class = np.argmax(predictions, axis=1).item()
#         with open("potato/class_indices.json") as f:
#             disease_mapping = json.load(f)       
#     elif id=="3" :
#         predictions = MODEL_tomato.predict(image_array)
#         predicted_class = np.argmax(predictions, axis=1).item()
#         with open("tomato/class_indices.json") as f:
#             disease_mapping = json.load(f)       

#     possibility = np.max(predictions) * 100
#     possibility = math.floor(possibility * 100) / 100
#     disease_name = disease_mapping.get(str(predicted_class), "Unknown disease")
#     if disease_name == "Healthy":
#         number_of_disease = 0
#     else :
#         number_of_disease = 1
#     return {
#         "number_of_disease":number_of_disease,
#         "result": [disease_name],
#         "possibility": [possibility]
#     }


# if __name__ == "__main__":
#     uvicorn.run(app,host='localhost',port=8000)


























# Final one---














# import math
# from PIL import Image
# from fastapi import FastAPI,File,UploadFile
# import uvicorn
# from pydantic import BaseModel
# import requests
# from fastapi.responses import FileResponse
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# import json
# from ultralytics import YOLO



# # # MODEL_rice = tf.keras.models.load_model("rice/MobileNet_rice.h5",compile=False)
# # # MODEL_potato = tf.keras.models.load_model("potato/MobileNet_potato.h5",compile=False)
# # MODEL_tomato = tf.keras.models.load_model("tomato/MobileNet_tomato.h5",compile=False)
# # Load the YOLOv8 model
# yolo_model = YOLO('yolov8n.pt')


# app = FastAPI()

# @app.get("/")
# async def start():
#     return "Welcome.."
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive..."


# # Directory to temporarily store the image
# UPLOAD_DIR = Path("downloads/")
# UPLOAD_DIR.mkdir(exist_ok=True)

# # structure of the incoming JSON request
# class ImageUrl(BaseModel):
#     imageUrl: str
#     modelId: str

# def preprocess_image(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     image_array = np.array(image) / 255.0  # Normalize
#     image_array = np.expand_dims(image_array, axis=0)  # batch dimension
#     return image_array

# async def download_image(url: str, file_location: Path) -> bool:
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         with open(file_location, "wb") as buffer:
#             for chunk in response.iter_content(1024):
#                 buffer.write(chunk)
#         return True
#     except requests.exceptions.RequestException as e:
#         print(f"Image not found")
#         return False
#     except Exception as e:
#         print(f"Image not found")
#         return False



# def is_relevant_image_yolo(image_path):
#     """
#     Checks if the image is relevant. Returns False if any object other than
#     potted plant (class ID 58) is detected with a confidence > 0.8.
#     Returns True otherwise (even if no potted plant is detected, as long as
#     no other object has high confidence).
#     """
#     try:
#         results = yolo_model.predict(image_path)

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 class_id = int(box.cls)
#                 confidence = float(box.conf)

#                 if confidence > 0.8 and class_id != 58:
#                     print(f"Other object detected with high confidence (> 0.7) (class {class_id}: {yolo_model.names[class_id]}). Image deemed not relevant.")
#                     return False  # Return False immediately if a highly confident non-potted plant is found
#         return True

#     except Exception as e:
#         print(f"Error processing image with YOLO: {e}")
#         return False

# @app.post("/image_predict")
# async def image_predict(image_url: ImageUrl):
#     url = image_url.imageUrl
#     id = image_url.modelId

#     # Check if the URL is valid by attempting to download the image
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise exception for bad responses
#     except requests.exceptions.RequestException as e:
#         return {"error": "image not found"}



#     file_name = "downloaded_image.jpg"  # Use a default name
#     file_location = UPLOAD_DIR / file_name

#     if not await download_image(url, file_location):
#         return {"error": "Unable to get the image reupload"}

#        # Check if the image is relevant
#     if not is_relevant_image_yolo(file_location):
#         return {
#             "number_of_disease": 0,
#             "result": ["Irrelevant"],
#             "possibility": [0.0]
#         }



#     image_array = preprocess_image(file_location)

#     # Make prediction
#     if id=="1" :        
#         MODEL_rice = tf.keras.models.load_model("rice/MobileNet_rice.h5",compile=False)
#         predictions = MODEL_rice.predict(image_array)
#         #get the class index or label
#         predicted_class = np.argmax(predictions, axis=1).item()  # Get the class index
#         # Load disease mapping from JSON file
#         with open("rice/class_indices.json") as f:
#             disease_mapping = json.load(f)
#     elif id=="2" :
#         MODEL_potato = tf.keras.models.load_model("potato/MobileNet_potato.h5",compile=False)
#         predictions = MODEL_potato.predict(image_array)
#         predicted_class = np.argmax(predictions, axis=1).item()
#         with open("potato/class_indices.json") as f:
#             disease_mapping = json.load(f)       
#     elif id=="3" :
#         MODEL_tomato = tf.keras.models.load_model("tomato/MobileNet_tomato.h5",compile=False)
#         predictions = MODEL_tomato.predict(image_array)
#         predicted_class = np.argmax(predictions, axis=1).item()
#         with open("tomato/class_indices.json") as f:
#             disease_mapping = json.load(f)
#     else : 
#         return {"error": "Invalid model ID"}

#     possibility = np.max(predictions) * 100
#     possibility = math.floor(possibility * 100) / 100
#     disease_name = disease_mapping.get(str(predicted_class), "Unknown disease")
#     if disease_name == "Healthy":
#         number_of_disease = 0
#     else :
#         number_of_disease = 1
#     return {
#         "number_of_disease":number_of_disease,
#         "result": [disease_name],
#         "possibility": [possibility]
#     }


# if __name__ == "__main__":
#     uvicorn.run(app,host='localhost',port=8000)







import math
from PIL import Image
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import requests
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
from ultralytics import YOLO

# Load models and mappings once at startup
yolo_model = None
try:
    yolo_model = YOLO('yolov8n.pt')  # Use the nano version of YOLO
except Exception as e:
    print(f"Error loading YOLO model: {e}")

MODEL_RICE = None
DISEASE_MAPPING_RICE = {}
try:
    MODEL_RICE = tf.keras.models.load_model("rice/MobileNet_rice.h5", compile=False)
    with open("rice/class_indices.json") as f:
        DISEASE_MAPPING_RICE = json.load(f)
except Exception as e:
    print(f"Error loading rice model: {e}")

MODEL_POTATO = None
DISEASE_MAPPING_POTATO = {}
try:
    MODEL_POTATO = tf.keras.models.load_model("potato/MobileNet_potato.h5", compile=False)
    with open("potato/class_indices.json") as f:
        DISEASE_MAPPING_POTATO = json.load(f)
except Exception as e:
    print(f"Error loading potato model: {e}")

MODEL_TOMATO = None
DISEASE_MAPPING_TOMATO = {}
try:
    MODEL_TOMATO = tf.keras.models.load_model("tomato/MobileNet_tomato.h5", compile=False)
    with open("tomato/class_indices.json") as f:
        DISEASE_MAPPING_TOMATO = json.load(f)
except Exception as e:
    print(f"Error loading tomato model: {e}")

app = FastAPI()

UPLOAD_DIR = Path("downloads/")
UPLOAD_DIR.mkdir(exist_ok=True)

class ImageUrl(BaseModel):
    imageUrl: str
    modelId: str

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

async def download_image(url: str, file_location: Path) -> bool:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_location, "wb") as buffer:
            for chunk in response.iter_content(1024):
                buffer.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Image not found: {e}")
        return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def is_relevant_image_yolo(image_path):
    if yolo_model is None:
        print("YOLO model not loaded, skipping relevance check.")
        return True
    try:
        results = yolo_model.predict(image_path)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                if confidence > 0.7 and class_id != 58:  # Reduced confidence threshold
                    print(
                        f"Other object detected with high confidence (> 0.7) (class {class_id}: {yolo_model.names[class_id]}). Image deemed not relevant."
                    )
                    return False
        return True
    except Exception as e:
        print(f"Error processing image with YOLO: {e}")
        return False

@app.get("/")
async def start():
    return "Welcome.."

@app.get("/ping")
async def ping():
    return "Hello, I am alive..."

@app.post("/image_predict")
async def image_predict(image_url: ImageUrl):
    url = image_url.imageUrl
    model_id = image_url.modelId

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": "image not found"}

    file_name = "downloaded_image.jpg"
    file_location = UPLOAD_DIR / file_name

    if not await download_image(url, file_location):
        return {"error": "Unable to download the image"}

    if yolo_model is not None and not is_relevant_image_yolo(file_location):
        return {
            "number_of_disease": 0,
            "result": ["Irrelevant"],
            "possibility": [0.0],
        }

    image_array = preprocess_image(file_location)
    predictions = None
    disease_mapping = {}
    model = None

    if model_id == "1":
        model = MODEL_RICE
        disease_mapping = DISEASE_MAPPING_RICE
    elif model_id == "2":
        model = MODEL_POTATO
        disease_mapping = DISEASE_MAPPING_POTATO
    elif model_id == "3":
        model = MODEL_TOMATO
        disease_mapping = DISEASE_MAPPING_TOMATO
    else:
        return {"error": "Invalid model ID"}

    if model is None:
        return {"error": f"Model with ID {model_id} failed to load."}

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1).item()
    possibility = np.max(predictions) * 100
    possibility = math.floor(possibility * 100) / 100
    disease_name = disease_mapping.get(str(predicted_class), "Unknown disease")
    number_of_disease = 0 if disease_name == "Healthy" else 1

    del image_array  # Explicitly delete image_array
    return {
        "number_of_disease": number_of_disease,
        "result": [disease_name],
        "possibility": [possibility],
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)








