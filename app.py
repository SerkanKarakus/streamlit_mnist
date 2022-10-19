import numpy as np
import cv2
import streamlit as st 
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
st.title("Mnist Digit Recognition")

col1, col2 = st.columns([1,1])
loaded_model=load_model('cnn-mnist-new.h5')

with col1:
    mode = st.checkbox("Draw or transform", True)
    canvas_result = st_canvas(
        fill_color = "#000000",
        stroke_width = 20,
        stroke_color = "#FFFFFF",
        background_color = "#000000",
        width = 256,
        height = 256,
        drawing_mode = "freedraw" if mode else "transform",
        key="canvas"
    
    )  
    pred_button = st.button("predict")
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data,(28,28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=0)
        test_img = img.reshape((1,28,28,1))/255

with col2:
    if pred_button:
        img_class = loaded_model.predict(test_img)
        probs = img_class
        fig = go.Figure(

                data = [
                        go.Bar(
                            x=np.arange(0,10),
                            y=(probs[0].tolist()*100)


                        )
                    ]
                )

        fig.update_layout(
                    width = 500,
                    height = 450
                )
        st.plotly_chart(fig)