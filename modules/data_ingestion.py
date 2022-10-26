import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time
import sys
import boto3
import io
import libs.feature_engineering as FE
import phik
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_option_menu import option_menu

def app():

    appSelectionSubCat = option_menu('Select option', ['Home','AWS(s3)','GCP','Azure','Google Analytics', 'API'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    # Sub pages -------------------------------------------------------------------------------------------------

    def appAWS():
        
        with st.expander("Get Data", expanded=False):

            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                aws_region_name = st.text_input("Region name")
            with col2:
                aws_access_key_id = st.text_input("Access key")
            with col3:
                aws_secret_access_key = st.text_input("Secret access")
            with col4:
                aws_bucket = st.text_input("Bucket name")

            col1, col2, col3 = st.columns([1.5,0.5,0.5])
            with col1:
                aws_prefix_folder = st.text_input("Input folder prefix")
            with col2:
                st.write(" ")
                st.write(" ")
                st.write(" ")
                aws_search_files = st.checkbox("Search files")

            if aws_search_files:

                s3 = boto3.resource('s3',
                    region_name=aws_region_name,
                    aws_access_key_id=aws_access_key_id, 
                    aws_secret_access_key=aws_secret_access_key)
                mybucket = s3.Bucket(aws_bucket)

                aws_list_objs = []
                for obj in mybucket.objects.filter(Prefix=aws_prefix_folder).all():
                    #print("File:" ,obj.key.split('/')[-1])
                    aws_list_objs.append(obj.key.split('/')[-1])
                aws_list_objs = [item for item in aws_list_objs if item != ""]
                #st.write(aws_list_objs)

                col1, col2, col3, col4 = st.columns([1.1,0.25,0.4,0.5])
                with col1:
                    select_aws_object = st.selectbox("Select the object", aws_list_objs)
                with col2:
                    type_object_to_save = st.selectbox("Type data", ['None','CSV','JSON','Parquet'])
                with col3:
                    if type_object_to_save == 'CSV':
                        type_comma = st.selectbox("Select separator",[',',';','|'])

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    name_dataset = st.text_input("Input dataset name")

                button_save_obj_aws_to_app = st.button("Apply", key='button_save_obj_aws_to_app')
                if button_save_obj_aws_to_app:
                    if type_object_to_save != 'None':
                        if type_object_to_save == 'CSV':
                            with st.spinner('Wait for it...'):
                                #s3_client = boto3.client('s3')
                                obj1 = s3.Bucket(aws_bucket).Object(f'{aws_prefix_folder}{select_aws_object}').get()
                                if not name_dataset in st.session_state['dataset']:
                                    if not name_dataset in st.session_state['Objects']:
                                        st.session_state[name_dataset] = pd.read_csv(io.BytesIO(obj1['Body'].read()), sep=type_comma)
                                        st.session_state['dataset'].append(name_dataset)
                                        st.session_state['Objects'].append(name_dataset)
                                        st.session_state['have_dataset'] = True
                                    else:
                                        st.warning("Objects name already exists")
                                else:
                                    st.warning("Dataset name already exists")
                        elif type_object_to_save == 'Parquet':
                            with st.spinner('Wait for it...'):
                                #s3_client = boto3.client('s3')
                                obj1 = s3.Bucket(aws_bucket).Object(f'{aws_prefix_folder}{select_aws_object}').get()
                                if not name_dataset in st.session_state['dataset']:
                                    if not name_dataset in st.session_state['Objects']:
                                        st.session_state[name_dataset] = pd.read_parquet(io.BytesIO(obj1['Body'].read()))
                                        st.session_state['dataset'].append(name_dataset)
                                        st.session_state['Objects'].append(name_dataset)
                                        st.session_state['have_dataset'] = True
                                    else:
                                        st.warning("Objects name already exists")
                                else:
                                    st.warning("Dataset name already exists")
                        else:
                            st.warning("Coming soon!")

                        st.success("File copied successfully")
                        #time.sleep(3)
                        #st.experimental_rerun()
            
                    else:
                        st.warning("Select object type")

        with st.expander("Upload Data", expanded=False):

            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                aws_region_name_upload = st.text_input("Region name", key="region_name_upload")
            with col2:
                aws_access_key_id_upload = st.text_input("Access key", key="acess_key_upload")
            with col3:
                aws_secret_access_key_upload = st.text_input("Secret access", key="secret_key_upload")
            with col4:
                aws_bucket_upload = st.text_input("Bucket name", key="bucket_name_upload")

            col1, col2 = st.columns([1.5,1.5])
            with col1:
                aws_prefix_folder_upload = st.text_input("Input folder prefix end file name - Ex: folder1/folder2/file.csv", key="upload_data")
            with col2:
                dataset_to_upload = st.selectbox("Select dataset", st.session_state['dataset'])

            if aws_region_name_upload and aws_access_key_id_upload and aws_secret_access_key_upload and aws_bucket_upload and dataset_to_upload != 'None':

                button_upload_s3 = st.button("Apply", key="button_upload_s3")
                if button_upload_s3:

                    with st.spinner('Wait for it...'):

                        s3_client = boto3.client(service_name='s3', region_name=aws_region_name_upload,
                            aws_access_key_id=aws_access_key_id_upload,
                            aws_secret_access_key=aws_secret_access_key_upload)

                        with io.StringIO() as csv_buffer:
        
                            st.session_state[dataset_to_upload].to_csv(csv_buffer, index=False)

                            response = s3_client.put_object(
                                Bucket=aws_bucket_upload, Key=aws_prefix_folder_upload, Body=csv_buffer.getvalue()
                            )
                            
                            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                            
                            if status == 200:
                                st.success(f"Successful S3 put_object response. Status - {status}")
                            else:
                                st.warning(f"Unsuccessful S3 put_object response. Status - {status}")

    def appGCP():
        st.warning('Coming soon!')

    def appAzure():
        st.warning('Coming soon!')

    def appGoogleAnalytics():
        st.warning('Coming soon!')

    def appAPI():
        st.warning('Coming soon!')


    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'AWS(s3)':
        appAWS()

    elif appSelectionSubCat == 'GCP':
        appGCP()

    elif appSelectionSubCat == 'Azure':
        appAzure()

    elif appSelectionSubCat == 'Google Analytics':
        appGoogleAnalytics()

    elif appSelectionSubCat == 'API':
        appAPI()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/data_ingestion.png'), width=700)

        