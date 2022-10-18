import streamlit as st
from PIL import Image
import base64
import time
from platform import python_version


def app():

    image = Image.open('images/accenture_ai.png')
    st.image(image, caption='Accenture AI',width=300)

    st.subheader("Data App")

    st.write("""
    *©Lucas Ferreira*
    """)

    st.write(f"Python version: {python_version()}")
    
    file_ = open("images/gif_coelho.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )






        

