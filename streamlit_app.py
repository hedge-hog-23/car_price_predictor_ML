# streamlit_app.py

import streamlit as st
import numpy as np  
from car_price_predictor import CarPricePredictor
import openai

# Create an instance of the CarPricePredictor
car_price_predictor = CarPricePredictor()

# Streamlit GUI
st.image('final.jpg', caption='CD and co', use_column_width=True)
st.sidebar.header("User Input")

# Dropdowns for user input
user_brand = st.sidebar.selectbox("Select car brand", car_price_predictor.data['brand'].unique())
# Filter models based on selected brand
filtered_models = car_price_predictor.data[car_price_predictor.data['brand'] == user_brand]['model'].unique()

# Dropdown for user input - Model
user_model = st.sidebar.selectbox("Select car model", filtered_models)
user_age = st.sidebar.number_input("Enter vehicle age", min_value=0.0, max_value=100.0, value=1.0)
user_km_driven = st.sidebar.number_input("Enter kilometers driven", min_value=0.0, value=10000.0)
user_fuel_type = st.sidebar.selectbox("Select fuel type", car_price_predictor.data['fuel_type'].unique())
user_transmission = st.sidebar.selectbox("Select transmission type", car_price_predictor.data['transmission_type'].unique())

# Predict button
if st.sidebar.button("Predict"):
    predicted_price = car_price_predictor.predict_price(user_brand, user_model, user_age, user_km_driven, user_fuel_type, user_transmission)
    st.success(f"Predicted Price: Rupees {np.round(predicted_price, 2)}")


# New section for user prompts
st.header("CD AI : Ask your doubts !!")

# Input box for user prompts
user_prompt = st.text_area("CD AI : Ask your doubts !!")

# OpenAI API key 
openai.api_key = ""

# Button to generate response using OpenAI API
if st.button("fetch results"):  
    if user_prompt:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150  # output length
            )

            # Display the generated response
            st.write("Generated Response:")
            st.write(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please enter a prompt before generating a response.")
