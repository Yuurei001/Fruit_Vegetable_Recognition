import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import requests
from bs4 import BeautifulSoup
import cv2

# Load model và định nghĩa labels
model = load_model('FV.keras')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def processed_img(img):
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = labels[y]
    return res.capitalize()

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        st.error("Failed to capture image")
        return None

def run_webcam():
    st.write("Click 'Capture' when ready to take a photo")
    if st.button('Capture'):
        img = capture_image()
        if img is not None:
            st.image(img, caption='Captured Image', use_column_width=True)
            st.write("Classifying...")
            result = processed_img(img)
            st.write(f"Prediction: {result}")
            if result.lower() in [v.lower() for v in vegetables]:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')

def run_image_upload():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        result = processed_img(img)
        st.write(f"Prediction: {result}")
        if result.lower() in [v.lower() for v in vegetables]:
            st.info('**Category : Vegetables**')
        else:
            st.info('**Category : Fruit**')
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + '(100 grams)**')

def main():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    
    option = st.radio(
        "Choose input method:",
        ('Upload Image', 'Use Webcam')
    )
    
    if option == 'Upload Image':
        run_image_upload()
    else:
        run_webcam()

if __name__ == "__main__":
    main()