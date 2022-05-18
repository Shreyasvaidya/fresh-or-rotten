from PIL import Image
import streamlit as st
import cv2
#import urllib.request
import numpy as np

st.title("Image Classification of select fruits and vegetables")
st.header("only orange,banana,apple,tomato,bitter guard, capsicum allowed at present")
st.text("Upload a clear image of fruit or vegetable ")
from image_classification import machine_classification 
uploaded_file = st.file_uploader("Enter image", type=["png","jpeg","jpg"])
#req = urllib.request.urlopen('https://drive.google.com/file/d/1xCW0kigCKAY6tPy1GN37JmbFM-b7HRr3/view?usp=sharing')
#arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#uploaded_file = cv2.imdecode(arr, -1) # 'Load it as it is'

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = machine_classification(image, 'ml_model.h5')
    if label>=0 and label<=5:
        st.write("The uploaded item is fresh")
    elif label>=6 and label<=11:
        st.write("The uploaded item is rotten")
    else:
        st.write("improper image or no image uploaded")
