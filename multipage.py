"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st
from streamlit_option_menu import option_menu

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        #self.pages.append({
          
        #        "title": title, 
        #        "function": func
        #    })

        self.pages.append({
          
                title : func
            })

    def run(self):
        #Drodown to select the page to run  
        #page = st.sidebar.selectbox(
        #    'App Navigation Menu', 
        #    self.pages, 
        #    format_func=lambda page: page['title']
        #)


        with st.sidebar:
            page = option_menu("App Navigation", list(map(lambda page: list(page.keys())[0], self.pages)),
                                icons=['house', 'gear', 'bi bi-share', 'bi bi-clipboard-data', 'bi bi-hammer', 'bi bi-stack', 'bi bi-magic'],
                                menu_icon="bi bi-box-arrow-right", default_index=0,
                                styles={"container": {"padding": "1!important", "background-color": "#fafafa"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}})

        #Run the app function 
        # page['function]()
        return [list(dic.values())[0] for dic in self.pages if list(dic.keys())[0] == page][0]()
