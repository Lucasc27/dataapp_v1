import streamlit as st
from PIL import Image
from persist import persist, load_widget_state
import base64
import pandas as pd

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Custom imports 
from multipage import MultiPage
from modules import data_preparation, home, data_settings, exploratory_data_analysis, feature_engineering, model_engineering, data_ingestion

# Configuração do aplicativo
image_icon = Image.open('images/icon.png')
PAGE_CONFIG = {'page_title':'Data App', 'page_icon':image_icon, 'layout':"wide"}
st.set_page_config(**PAGE_CONFIG)

def main():
    if "appSelection" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "appSelection": "Home",
            # Default widget values.
            "Objects": [], # Lista de objetos para criar datasets
            "dataset": [], # Lista de datasets carregados
            #"df":pd.read_csv('base_input/df_com_caracteres_especiais.csv'),
            "Variables": [],
            "have_dataset": False
        })

# -----------------------------------------------------------------------------------------------------------------------
# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Data Settings", data_settings.app)
app.add_page("Data Ingestion",data_ingestion.app)
app.add_page("Exploratory Data Analysis", exploratory_data_analysis.app)
app.add_page("Data Preparation",data_preparation.app)
app.add_page("Feature Engineering", feature_engineering.app)
app.add_page("Model Engineering",model_engineering.app)


# The main app
app.run()

# -----------------------------------------------------------------------------------------------------------------------


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if __name__ == "__main__":
    load_widget_state()
    main()
