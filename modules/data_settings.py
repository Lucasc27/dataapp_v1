import streamlit as st
from PIL import Image
import pandas as pd
import time
import sys
import io
import os
import zipfile
from streamlit_option_menu import option_menu

def app():
    
    appSelectionSubCat = option_menu('Select option', ['Home','Load Datasets','View Dataset','View Variables','Delete dataset/variable',
                                            'System Variables Size','Variables Optimization','Set Dtypes','Copy or Rename Obj',
                                            'Download Dataset','Download Output Files'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    #appSelectionSubCat = st.sidebar.selectbox('Submenu',['Home','Load Datasets','View Bases','Delete Bases',
    #                                        'System Variables Size','Variables Optimization','Set Dtypes','Copy Objects',
    #                                        'Download Dataset'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def page_settings_load_dataset():

        optionDataset = st.selectbox(
        'Select an object to save a dataset',
        (st.session_state['Objects']))

        if optionDataset:

                myBase = st.radio("Select an option",('Upload csv','Upload xlsx', 'Path from csv'))
                with st.form(key='my_form_load_dataset', clear_on_submit=True):
                    if not optionDataset in st.session_state['dataset']:
                        if myBase == 'Upload csv':

                            check_delimiter = st.checkbox("Custom Delimiter")
                            deLim = st.text_input('Input the delimiter')
                            check_parse_dates = st.checkbox("Custom date type")
                            var_parse_dates = st.text_input("Input the variables")
                            var_parse_dates = var_parse_dates.split(',')
                            
                            if check_delimiter:
                                uploaded_file = st.file_uploader("Choose a file")
                                if uploaded_file is not None:

                                    bytes_data = uploaded_file.getvalue()

                                    if check_parse_dates:
                                        df_default = pd.read_csv(uploaded_file, sep=str(deLim), parse_dates=var_parse_dates, date_parser=lambda col: pd.to_datetime(col))
                                    else:
                                        df_default = pd.read_csv(uploaded_file, sep=str(deLim))

                                    st.session_state[optionDataset] = df_default
                                    st.session_state['dataset'].append(optionDataset)
                                    st.session_state['have_dataset'] = True
                            else:
                                uploaded_file = st.file_uploader("Choose a file")
                                if uploaded_file is not None:

                                    bytes_data = uploaded_file.getvalue()

                                    if check_parse_dates:
                                        df_default = pd.read_csv(uploaded_file, parse_dates=var_parse_dates, date_parser=lambda col: pd.to_datetime(col))
                                    else:
                                        df_default = pd.read_csv(uploaded_file)

                                    st.session_state[optionDataset] = df_default
                                    st.session_state['dataset'].append(optionDataset)
                                    st.session_state['have_dataset'] = True

                        elif myBase == 'Upload xlsx':
                                
                            uploaded_file = st.file_uploader("Choose a file")
                            if uploaded_file is not None:

                                bytes_data = uploaded_file.getvalue()

                                df_default = pd.read_excel(uploaded_file)
                                st.session_state[optionDataset] = df_default
                                st.session_state['dataset'].append(optionDataset)
                                st.session_state['have_dataset'] = True

                        elif myBase == 'Path from csv':

                            pathBase = st.text_input('*Input the base path')

                            check_custom_csv = st.checkbox("Custom")
                            deLim = st.text_input('Input the delimiter')
                            headerSelectedCSV = st.number_input("Input the number of header", min_value=0, value=1)
                            columnSelectedCSV = st.text_input("Input the columns")
                            nrowsSelectedCSV = st.number_input("Input the number of rows", min_value=0, value=5)

                            check_parse_dates = st.checkbox("Custom date type")
                            var_parse_dates = st.text_input("Input the variables")
                            var_parse_dates = var_parse_dates.split(',')

                            columnSelectedCSV2 = columnSelectedCSV.split(',')

                            if pathBase:
                                if check_custom_csv:
                                    if check_parse_dates:
                                        df_default = pd.read_csv(pathBase, sep=str(deLim) if deLim else None , header=int(headerSelectedCSV), nrows=int(nrowsSelectedCSV) if nrowsSelectedCSV else None, names=columnSelectedCSV2 if columnSelectedCSV2 else None, usecols=columnSelectedCSV2 if columnSelectedCSV2 else None, parse_dates=var_parse_dates, date_parser=lambda col: pd.to_datetime(col))
                                    else:
                                        df_default = pd.read_csv(pathBase, sep=str(deLim) if deLim else None , header=int(headerSelectedCSV), nrows=int(nrowsSelectedCSV) if nrowsSelectedCSV else None, names=columnSelectedCSV2 if columnSelectedCSV2 else None, usecols=columnSelectedCSV2 if columnSelectedCSV2 else None)
                                    st.session_state[optionDataset] = df_default
                                    st.session_state['dataset'].append(optionDataset)
                                    st.session_state['have_dataset'] = True
                                else:
                                    if check_parse_dates:
                                        df_default = pd.read_csv(pathBase, parse_dates=var_parse_dates, date_parser=lambda col: pd.to_datetime(col))
                                    else:
                                        df_default = pd.read_csv(pathBase)
                                    st.session_state[optionDataset] = df_default
                                    st.session_state['dataset'].append(optionDataset)
                                    st.session_state['have_dataset'] = True

                    else:
                        st.write("<u><b>Dataset loaded</b></u>", unsafe_allow_html=True)

                    submit = st.form_submit_button('Submit')
                    if submit:
                        st.experimental_rerun()

        if st.session_state['have_dataset']:

            SelectionHeadDataset = st.selectbox(
            'Select a base', (st.session_state['dataset']))

            showbase = st.checkbox('Preview')
            if showbase:
                st.dataframe(st.session_state[SelectionHeadDataset].head())
                st.dataframe(st.session_state[SelectionHeadDataset].tail())
        else:
            st.write("There is no Dataset loaded")
 
    def page_settings_delete_dataset():

        with st.expander("Delete dataset or variable",expanded=True):
            
            col1, col2 = st.columns([1.5,3.5])
            with col1:
                variable_or_dataset_delete = st.selectbox("Object to delete is a dataset or variable?",['None','Dataset','Variable'])
            
            if variable_or_dataset_delete == 'Dataset':
                SelectionToDelete = st.selectbox(
                'Select a base', (st.session_state['dataset']))
            elif variable_or_dataset_delete == 'Variable':
                SelectionToDelete = st.selectbox(
                'Select a base', (st.session_state['Variables']))

            with st.form(key='my_form_delete_dataset', clear_on_submit=True):

                st.write("To delete the object, answer '<u><em>permanently delete</u></em>'.", unsafe_allow_html=True)
                del_dataset = st.text_input("Answer and submit:")

                submit = st.form_submit_button('Submit')

                if submit:
                    if SelectionToDelete:
                        if del_dataset == 'permanently delete':
                            if variable_or_dataset_delete == 'Dataset':
                                del st.session_state[SelectionToDelete]
                                st.session_state['dataset'].remove(SelectionToDelete)
                                st.session_state['Objects'].remove(SelectionToDelete)
                                st.session_state['have_dataset'] = True if (len(st.session_state['dataset']) > 0) else False
                                st.success('Dataset successfully deleted!')
                                time.sleep(1)
                                st.experimental_rerun()
                            if variable_or_dataset_delete == 'Variable':
                                del st.session_state[SelectionToDelete]
                                st.session_state['Variables'].remove(SelectionToDelete)
                                st.session_state['Objects'].remove(SelectionToDelete)
                                st.success('Dataset successfully deleted!')
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                st.error('Select the option to delete!')
                                time.sleep(1)
                                st.experimental_rerun()
                        else:
                            st.error('Sentence incorrectly!')
                            time.sleep(1)
                            st.experimental_rerun()
                    else:
                        st.error('Select the object!')
                        time.sleep(1)
                        st.experimental_rerun()

    def page_settings_view_base():

        optionDatasetView = st.selectbox(
        'Select a dataset',
        (st.session_state['dataset']))

        with st.expander("",expanded=True):
        
            if st.session_state['have_dataset']:

                col1, col2, col3 = st.columns([0.8,0.8,2.5])
                with col1:
                    init_rows = st.number_input("Min. of rows", value=0, min_value=0)
                with col2:
                    #final_rows = st.number_input("Max. of rows", value=5, min_value=0, max_value=int(st.session_state[optionDatasetView].shape[0]))
                    final_rows = st.number_input("Max. of rows", value=st.session_state[optionDatasetView].shape[0] if st.session_state[optionDatasetView].shape[0] < 5 else 5, min_value=0, max_value=int(st.session_state[optionDatasetView].shape[0]))

                col11, col22, col33 = st.columns([0.8,0.8,2.5])
                with col11:
                    init_cols = st.number_input("Min. of cols", value=0, min_value=0)
                with col22:
                    final_cols = st.number_input("Max. of cols", value=int(st.session_state[optionDatasetView].shape[1]), min_value=0, max_value=int(st.session_state[optionDatasetView].shape[1]))

            else:
                st.write("There is no Dataset loaded")

        selectionColumn = st.checkbox('Columns selections')

        if st.session_state.have_dataset:

            if selectionColumn:
                optionsViewColumns = st.multiselect("Multiselect columns",list(st.session_state[optionDatasetView].columns))
                #st.write(optionsViewColumns)
                st.write(
                    f"""
                    **Columns selected**: {", ".join(optionsViewColumns)}
                    """
                )
                st.dataframe(st.session_state[optionDatasetView][optionsViewColumns].iloc[int(init_rows):int(final_rows)])
            else:
                st.dataframe(st.session_state[optionDatasetView].iloc[int(init_rows):int(final_rows), int(init_cols):int(final_cols)])

    def page_settings_view_variables():

        if st.session_state['Variables']:

            optionVariableView = st.selectbox(
            'Select a variables',
            (st.session_state['Variables']))

            st.write(st.session_state[optionVariableView])



        else:
            st.write("There is no variable loaded")

    def page_settings_system_variables_size():
        
        #list_size_of = {}
        #for var, obj in st.session_state.items():
        #    list_size_of[var] = sys.getsizeof(obj)
        def sizeof_fmt(num):
            #for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            for unit in ['','KByte','MByte','GByte','TByte','PByte','EByte','ZByte']:
                if abs(num) < 1024.0:
                    return "%3.1f %s" % (num, unit)
                num /= 1024.0
            return "%.1f %s" % (num, 'Yi')

        name_var_sizeof = []
        tam_var_sizeof = []
        df_sizeof = pd.DataFrame(columns=["Variable","Size"])
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in st.session_state.items()),
                                key= lambda x: -x[1]):
            name_var_sizeof.append(name)
            tam_var_sizeof.append(sizeof_fmt(size))

        df_sizeof['Variable'] = name_var_sizeof
        df_sizeof['Size'] = tam_var_sizeof
        df_sizeof_fim = df_sizeof[(df_sizeof['Variable'] != 'have_dataset') & (df_sizeof['Variable'] != 'dataset') & (df_sizeof['Variable'] != 'Objects') & (df_sizeof['Variable'] != 'Variables') & (df_sizeof['Variable'] != 'appSelection')]
        st.write(df_sizeof_fim)
        
    def page_settings_variables_optimization():

        optionDatasetSelect = st.selectbox(
        'Select a dataset',
        (st.session_state['dataset']))

        #with st.expander("",expanded=True):
        
        if st.session_state['have_dataset']:

            with st.form(key='my_form_optimization_variables', clear_on_submit=True):

                @st.cache
                def get_info(df):

                    dfx_dtypes = pd.DataFrame(df.dtypes.values, columns=['type'])
                    dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                    x_dtypes = list(dfx_dtypes['type_str'].values)

                    def sizeof_fmt(num):
                        #for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
                        for unit in ['','KByte','MByte','GByte','TByte','PByte','EByte','ZByte']:
                            if abs(num) < 1024.0:
                                return "%3.1f %s" % (num, unit)
                            num /= 1024.0
                        return "%.1f %s" % (num, 'Yi')

                    # Salvando as colunas do dataframe em um dicionário
                    dic_df = {}
                    for key in df:
                        dic_df[key] = df[key]
                    
                    name_var_sizeof = []
                    tam_var_sizeof = []
                    df_sizeof = pd.DataFrame(columns=["Variable","Size"])

                    for name, size in (((name, sys.getsizeof(value)) for name, value in dic_df.items())):
                        name_var_sizeof.append(name)
                        tam_var_sizeof.append(sizeof_fmt(size))
                        
                    df_sizeof['Variable'] = name_var_sizeof
                    df_sizeof['Size'] = tam_var_sizeof

                    #return pd.DataFrame({'Objects Types': x_dtypes, 'Size': df_sizeof['Size'].values})
                    return pd.DataFrame({'Objects Types': x_dtypes, 'Size': df_sizeof['Size'].values, 'NaN': df.isna().sum(), 'NaN%': round((df.isna().sum()/len(df))*100,2), 'Unique':df.nunique()})



                from typing import List
                def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
                    floats = df.select_dtypes(include=['float64','float32']).columns.tolist()
                    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
                    return df
                
                def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
                    ints = df.select_dtypes(include=['int64']).columns.tolist()
                    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')

                    return df
                
                def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
                    for col in df.select_dtypes(include=['object']):
                        if col not in datetime_features:
                            if not (type(df[col][0])==list):
                                num_unique_values = len(df[col].unique())
                                num_total_values = len(df[col])
                                if float(num_unique_values) / num_total_values < 0.5:
                                    df[col] = df[col].astype('category')
                        else:
                            df[col] = pd.to_datetime(df[col])

                    return df
                
                def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
                    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))
                    #return optimize_ints(optimize_objects(df, datetime_features), datetime_features)

                check_var_datetime = st.checkbox("Datetime variable")
                
                datatimeColumnOptimization = st.multiselect("Select columns for transformation into datatime",list(st.session_state[optionDatasetSelect].columns))

                removeColumnOptimization = st.multiselect("Select columns to remove from optimization",list(st.session_state[optionDatasetSelect].columns))

                addColumnOptimization = st.multiselect("Select columns to add in optimization",list(st.session_state[optionDatasetSelect].columns))
                
                submit = st.form_submit_button('Submit')

                if submit:
                    if check_var_datetime:
                        if datatimeColumnOptimization:
                            if (removeColumnOptimization) and (not addColumnOptimization):
                                df_columns_to_exit = st.session_state[optionDatasetSelect][removeColumnOptimization]
                                st.session_state[optionDatasetSelect] = optimize(st.session_state[optionDatasetSelect].drop(list(removeColumnOptimization),axis=1), datatimeColumnOptimization)
                                st.session_state[optionDatasetSelect] = pd.concat([df_columns_to_exit, st.session_state[optionDatasetSelect]],axis=1)
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                            elif (addColumnOptimization) and (not removeColumnOptimization):
                                df_columns_to_add_concat = st.session_state[optionDatasetSelect].drop(addColumnOptimization, axis=1)
                                df_columns_to_add_concat = df_columns_to_add_concat.drop(datatimeColumnOptimization, axis=1)

                                df_opt = pd.DataFrame()
                                if len(addColumnOptimization) > 1:
                                    for col in addColumnOptimization:
                                        df_opt[col] = st.session_state[optionDatasetSelect][col]
                                else:
                                    df_opt[addColumnOptimization] = st.session_state[optionDatasetSelect][addColumnOptimization]
                                
                                if len(datatimeColumnOptimization) > 1:
                                    for col in datatimeColumnOptimization:
                                        df_opt[col] = st.session_state[optionDatasetSelect][col]
                                else:
                                    df_opt[datatimeColumnOptimization] = st.session_state[optionDatasetSelect][datatimeColumnOptimization]
                                
                                
                                df_opt_out = optimize(df_opt, datatimeColumnOptimization)
                                st.session_state[optionDatasetSelect] = pd.concat([df_opt_out,df_columns_to_add_concat],axis=1)
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                            elif (removeColumnOptimization) and (addColumnOptimization):
                                st.error("Select multiple columns or delete multiple columns")
                                time.sleep(2)
                                st.experimental_rerun()
                            else:
                                st.session_state[optionDatasetSelect] = optimize(st.session_state[optionDatasetSelect], datatimeColumnOptimization)
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                        else:
                            st.error("Select a column of datatime")
                            time.sleep(2)
                            st.experimental_rerun()
                        
                    else:
                        if not datatimeColumnOptimization:
                            if (removeColumnOptimization) and (not addColumnOptimization):
                                df_columns_to_exit = st.session_state[optionDatasetSelect][removeColumnOptimization]
                                st.session_state[optionDatasetSelect] = optimize(st.session_state[optionDatasetSelect].drop(list(removeColumnOptimization),axis=1))
                                st.session_state[optionDatasetSelect] = pd.concat([df_columns_to_exit, st.session_state[optionDatasetSelect]],axis=1)
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                            elif (addColumnOptimization) and (not removeColumnOptimization):
                                df_columns_to_add_concat = st.session_state[optionDatasetSelect].drop(addColumnOptimization, axis=1)
                                st.session_state[optionDatasetSelect] = optimize(st.session_state[optionDatasetSelect][addColumnOptimization])
                                st.session_state[optionDatasetSelect] = pd.concat([st.session_state[optionDatasetSelect],df_columns_to_add_concat],axis=1)
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                            elif (removeColumnOptimization) and (addColumnOptimization):
                                st.error("Select multiple columns or delete multiple columns")
                                time.sleep(2)
                                st.experimental_rerun()
                            else:
                                st.session_state[optionDatasetSelect] = optimize(st.session_state[optionDatasetSelect])
                                st.success("Optimization done successfully")
                                time.sleep(2)
                                st.experimental_rerun()
                        else:
                            st.error("Select datetime check")
                            time.sleep(1)
                            st.experimental_rerun()
            
        else:
            st.write("There is no Dataset loaded")

        if st.session_state['have_dataset']:

            st.dataframe(st.session_state[optionDatasetSelect].head())

            col1, col2 = st.columns([2.5,1.5])

            with col1:

                st.dataframe(get_info(st.session_state[optionDatasetSelect]))
                
            with col2:

                buffer = io.StringIO() 
                st.session_state[optionDatasetSelect].info(buf=buffer, verbose=False, memory_usage='deep')
                s = buffer.getvalue()
                st.text(s)
            
    def page_settings_copy_rename_objects():

        st.title("Copy or rename objects")

        optionDataset = st.selectbox(
        'Select the variable to copy or rename',
        (st.session_state['Objects']))

        if optionDataset:

            with st.expander("Copy", expanded=False):

                with st.form(key='my_form_copy_object', clear_on_submit=True):

                    optionObjects = st.selectbox(
                    'Select an empty object',
                    (st.session_state['Objects']))

                    col1, col2 = st.columns([1,4])
                    with col1:
                        variable_or_dataset_rename = st.selectbox("Object is dataset or variable?",['None','Dataset','Variable'])

                    st.write("To copy a dataset in another object, answer '<u><em>copy</u></em>'.", unsafe_allow_html=True)
                    copy_dataset = st.text_input("Answer and submit:")

                    submit_copy = st.form_submit_button('Submit')

                    if submit_copy:
                        if copy_dataset == 'copy':
                            if variable_or_dataset_rename == 'Dataset':
                                if not optionObjects in st.session_state['dataset']:
                                    st.session_state[optionObjects] = st.session_state[optionDataset].copy(deep=True)
                                    st.session_state['dataset'].append(optionObjects)
                                    st.success("Copy created successfully!")
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    st.error('These objects cannot be changed.')
                                    time.sleep(1)
                                    st.experimental_rerun()
                            elif variable_or_dataset_rename == 'Variable':
                                if not optionObjects in st.session_state['Variables']:
                                    st.session_state[optionObjects] = st.session_state[optionDataset]
                                    st.session_state['Variable'].append(optionObjects)
                                    st.success("Copy created successfully!")
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    st.error('These objects cannot be changed.')
                                    time.sleep(1)
                                    st.experimental_rerun()
                            else:
                                st.error('Select the type of object!')
                                time.sleep(1)
                                st.experimental_rerun()
                            
                        else:
                            st.error('Sentence incorrectly!')
                            time.sleep(1)
                            st.experimental_rerun()

            with st.expander("Rename", expanded=False):

                with st.form(key='my_form_rename_object', clear_on_submit=True):

                    new_name_object = st.text_input("Input the new object name")
                    col1, col2 = st.columns([1,4])
                    with col1:
                        variable_or_dataset_rename = st.selectbox("Object is dataset or variable?",['None','Dataset','Variable'])
                    submit_rename = st.form_submit_button('Submit')

                    if submit_rename:
                        if new_name_object != '' and variable_or_dataset_rename != 'None':
                            if variable_or_dataset_rename == 'Dataset':
                                st.session_state[new_name_object] = st.session_state[optionDataset]
                                st.session_state['dataset'].append(new_name_object)
                                st.session_state['Objects'].append(new_name_object)
                                del st.session_state[optionDataset]
                                st.session_state['dataset'].remove(optionDataset)
                                st.session_state['Objects'].remove(optionDataset)
                                st.success("Renamed!")
                                time.sleep(2)
                                st.experimental_rerun()
                            elif variable_or_dataset_rename == 'Variable':
                                st.session_state[new_name_object] = st.session_state[optionDataset]
                                st.session_state['Variables'].append(new_name_object)
                                st.session_state['Objects'].append(new_name_object)
                                del st.session_state[optionDataset]
                                st.session_state['Variables'].remove(optionDataset)
                                st.session_state['Objects'].remove(optionDataset)
                                st.success("Renamed!")
                                time.sleep(2)
                                st.experimental_rerun()
                            else:
                                st.error('Select the type of object!')
                                time.sleep(2)
                                st.experimental_rerun()

        else:
            st.write("There is no Dataset loaded")

    def page_settings_download_dataset():
        
        optionDataset = st.selectbox(
        'Select the variable',
        #(st.session_state['dataset']))
        (st.session_state['dataset']))

        if optionDataset:

            with st.expander("Download dataset",expanded=True):
                
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode('utf-8')

                name_csv = st.text_input("Input the file name with the '.csv'", value=str(optionDataset) + '.csv')

                if name_csv:
                    st.download_button(
                        label="Download data as CSV",
                        data=convert_df(st.session_state[optionDataset]),
                        file_name=name_csv,
                        mime='text/csv',
                    )
                else:
                    st.error("Write the name of file!")

        else:
            st.write("There is no Dataset loaded")

    def page_settings_set_dtypes():

        from datetime import datetime

        optionDataset = st.selectbox(
        'Select the variable',
        (st.session_state['dataset']))

        if optionDataset:

            with st.expander("Modify data types",expanded=True):

                def get_info_dtypes(df):
                    dfx_dtypes = pd.DataFrame(df.dtypes.values, columns=['type'])
                    dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                    x_dtypes = list(dfx_dtypes['type_str'].values)
                    return pd.DataFrame({'Objects Types': x_dtypes, 'NaN': df.isna().sum()})

                col1, col2 = st.columns([2,1])
                
                with col1:
                    
                    with st.form(key='my_form_transform_dtypes', clear_on_submit=True):

                        st.write(" ")

                        col_to_change = st.multiselect("Select the columns to be transformed", st.session_state[optionDataset].columns)

                        st.write(" ")
                        st.write(" ")

                        select_dtype = st.selectbox("Select the type:",['None','Integer','Float','Object', 'Category', 'Datetime'])
                        
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        submit = st.form_submit_button('Submit')

                        if submit:

                            if select_dtype == 'Integer':
                                st.session_state[optionDataset][col_to_change] = st.session_state[optionDataset][col_to_change].astype(int)
                                st.success("Modified type")
                            elif select_dtype == 'Float':
                                st.session_state[optionDataset][col_to_change] = st.session_state[optionDataset][col_to_change].astype(float)
                                st.success("Modified type")
                            elif select_dtype == 'Object':
                                st.session_state[optionDataset][col_to_change] = st.session_state[optionDataset][col_to_change].astype(object)
                                st.success("Modified type")
                            elif select_dtype == 'Category':
                                st.session_state[optionDataset][col_to_change] = st.session_state[optionDataset][col_to_change].astype("category")
                                st.success("Modified type")
                            elif select_dtype == 'Datetime':
                                for c in col_to_change:
                                    st.session_state[optionDataset][c]=pd.to_datetime(st.session_state[optionDataset][c].astype(str), format='%Y-%m-%d')
                                st.success("Modified type")

                            else:
                                st.error("Select an type")

                            time.sleep(0.1)
                            st.experimental_rerun()

                with col2:

                    st.dataframe(get_info_dtypes(st.session_state[optionDataset]), height=355)

                st.subheader("Preview dataset")
                st.dataframe(st.session_state[optionDataset].head())
                
        else:
            st.write("There is no Dataset loaded")

    def page_settings_download_output_files():

        with st.expander("Download output files", expanded=False):

            for _, _, arquivos in os.walk('files_output/'):
                list_files = arquivos
            list_files.insert(0,'None')

            options_files = st.selectbox('Select file', list_files, key="options_files")
            name_file = st.text_input("File name")
            check_convert = st.checkbox("Convert")

            if options_files != 'None' and name_file != '' and check_convert:

                def zipp():
                    zipfile.ZipFile(f"files_output/{name_file}.zip", mode="w").write(f"files_output/{options_files}")
                    with open(f"files_output/{name_file}.zip", "rb") as fp:
                        file_zip_contents = fp.read()
                    return file_zip_contents

                st.download_button(
                    label="Download",
                    data=zipp(),
                    file_name=name_file+'.zip',
                    mime="application/zip"
                )

        with st.expander("Delete output files", expanded=False):

            for _, _, arquivos in os.walk('files_output/'):
                list_files = arquivos
            list_files.insert(0,'None')

            options_files_delete = st.selectbox('Select file', list_files, key="options_files_delete")

            delete_button_output = st.button("Delete")
            if delete_button_output:
                os.remove(f'files_output/{options_files_delete}')
                st.success("File deleted successfully")
                time.sleep(2)
                st.experimental_rerun()

    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Load Datasets':
        page_settings_load_dataset()

    elif appSelectionSubCat == 'View Dataset':
        page_settings_view_base()

    elif appSelectionSubCat == 'View Variables':
        page_settings_view_variables()

    elif appSelectionSubCat == 'Delete dataset/variable':
        page_settings_delete_dataset()

    elif appSelectionSubCat == 'System Variables Size':
        page_settings_system_variables_size()

    elif appSelectionSubCat == 'Variables Optimization':
        page_settings_variables_optimization()

    elif appSelectionSubCat == 'Set Dtypes':
        page_settings_set_dtypes()

    elif appSelectionSubCat == 'Copy or Rename Obj':
        page_settings_copy_rename_objects()

    elif appSelectionSubCat == 'Download Dataset':
        page_settings_download_dataset()

    elif appSelectionSubCat == 'Download Output Files':
        page_settings_download_output_files()

    elif appSelectionSubCat == 'Home':

        #st.image(Image.open('images/image6.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        General Settings
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )

        with st.expander("Create or deleted object for dataset", expanded=False):

            with st.form(key='my_form_append_objects', clear_on_submit=True):

                st.write("Add object")
                value_objects = st.text_input("Name:")
                submit = st.form_submit_button('Submit')

                if submit:
                    if value_objects not in st.session_state['Objects']:
                        if value_objects not in st.session_state:
                            st.success('Successfully added '+ str(value_objects) + ' object!')
                            st.session_state['Objects'].append(value_objects)
                            time.sleep(0.1)
                            st.experimental_rerun()
                        else:
                            st.error(f"{value_objects} name is reserved for system variable")
                            time.sleep(0.1)
                            st.experimental_rerun()
                    else:
                        st.error("Object exists in the list")
                        time.sleep(0.1)
                        st.experimental_rerun()

            with st.form(key='my_form_remove_objects', clear_on_submit=True):

                st.write("Remove object")
                value_objects = st.text_input("Name:")
                submit = st.form_submit_button('Submit')

                if submit:
                    if (str(value_objects) in st.session_state['Objects']):
                        st.success('['+str(value_objects)+']' + ' object removed!')
                        st.session_state['Objects'].remove(value_objects)
                        #del st.session_state[value_objects]
                        time.sleep(0.1)
                        st.experimental_rerun()
                    else:
                        st.error("Object does not exist in the list")
                        time.sleep(0.1)
                        st.experimental_rerun()

            with st.form(key='my_form_rename_objects', clear_on_submit=True):

                st.write("Rename object")
                name_object = st.selectbox("Select object",st.session_state['Objects'])
                newName_object = st.text_input("New object name:")
                submit = st.form_submit_button('Submit')

                if submit:
                    if not name_object in st.session_state['dataset']:
                        if not name_object in st.session_state['Variables']:
                            st.session_state['Objects'].remove(name_object)
                            st.session_state['Objects'].append(newName_object)
                            st.success('successfully renamed!')
                            time.sleep(1)
                            st.experimental_rerun()
                        else:
                            st.warning('Object used in the variable list!')
                            time.sleep(1)
                            st.experimental_rerun()
                    else:
                            st.warning('Object used in the dataset list!')
                            time.sleep(1)
                            st.experimental_rerun()

            st.write("Objects saved in the system", st.session_state['Objects'])