import keras
import streamlit as st
from PIL import Image
from PIL import ImageOps
import cv2
import numpy as np
from rembg import remove

def machine_classification(img,weights_file ):
    # Load the model
    @st.cache_resource
    def load_model_catche(model):
        return keras.models.load_model(model)
    model = load_model_catche(weights_file)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 45, 45, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (45, 45)
    image = ImageOps.fit(image, size,Image.LANCZOS)
    image=remove(image)
    try:
        image.save("geeks.jpeg")
	 #turn the image into a numpy array
        img_array=np.array(cv2.imread("geeks.jpeg"))
    except:
        image.save("geeks.png")
        img_array=np.array(cv2.imread("geeks.png"))    
   
  
        

    array=cv2.resize(img_array,size)
    # Normalize the image
    normalized_image_array = array/255 

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability
