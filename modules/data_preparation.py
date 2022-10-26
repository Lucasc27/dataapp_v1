import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import libs.EDA_graphs as EDA
import time
#import libs.model_engineering_new as me
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pickle
from streamlit_ace import st_ace
import re
import os
from streamlit_option_menu import option_menu



def app():

    # Atualizando as variáveis de sessão
    list_to_objects = []
    list_to_datasets = []
    list_to_add = []
    for item in st.session_state.keys():
        if not type(st.session_state[item]) == bool:
            if not item in ['Objects','dataset','appSelection','Variables']:
                list_to_add.append(item)

    for item in list_to_add:
        if not item in st.session_state['Objects']:
            st.session_state['Objects'].append(item)
    for item in list_to_add:
        if not item in st.session_state['dataset'] and type(st.session_state[item]) == pd.DataFrame:
            st.session_state['dataset'].append(item)

    appSelectionSubCat = option_menu('Select option', ['Home','Automated Pre-processing','Execute Script in Object', 'Execute Script in Session'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    #appSelectionSubCat = st.sidebar.selectbox('Submenu',['Home','Automated Pre-processing','Execute Script .py'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appAutomatedPreProcessing():

        st.title('Automated Pre-Processing')

        with st.expander("Click here for more info on this app section", expanded=False):

            st.write(
            f"""

            Summary
            ---------------
            Welcome to the application pre-processing session. Pre-processing is a very important step in AI model 
            development or data analysis. This app provides a quick way to use preprocessing methods on your dataset.

            **select the dataset and proceed with the execution**
            """
            )
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        button_update_code_script = st.button("Refresh page", key='button_automatedpreprocessing_code_script')
        if button_update_code_script:
            st.experimental_rerun()

        if optionDataset:

            dataset_columns = st.session_state[optionDataset].columns
            options_columns = dataset_columns.insert(0, 'None')

            # ----------------------------- AQUI
            st.subheader("Select the type processing")
            # -----------------------------
            
            with st.expander("Remove special characters", expanded=False):

                submit_remove_caracters = st.button("Apply", key='apply_remove_caracters')

                code_remove_caracters = \
f'''# import the lib -> import unidecode
df_cols = pd.DataFrame(self.base.columns, columns=['cols'])
df_cols['cols'] = df_cols['cols'].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True)
self.base.columns = df_cols['cols'].values

for c in self.base.select_dtypes(include=['object','category']).columns:
    self.base[c] = self.base[c].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True).str.lower()
for c in self.base.select_dtypes(include=['object','category']).columns:
    self.base[c] = self.base[c].apply(lambda x: unidecode.unidecode(str(x)))
'''
                st.code(code_remove_caracters, language='python')
                if submit_remove_caracters:
                    import unidecode
                    df_cols = pd.DataFrame(st.session_state[optionDataset].columns, columns=['cols'])
                    df_cols['cols'] = df_cols['cols'].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True)
                    st.session_state[optionDataset].columns = df_cols['cols'].values

                    for c in st.session_state[optionDataset].select_dtypes(include=['object','category']).columns:
                        st.session_state[optionDataset][c] = st.session_state[optionDataset][c].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True).str.lower()
                    for c in st.session_state[optionDataset].select_dtypes(include=['object','category']).columns:
                        st.session_state[optionDataset][c] = st.session_state[optionDataset][c].apply(lambda x: unidecode.unidecode(str(x)))
                    st.experimental_rerun()

            with st.expander("Rename, copy, keep and delete variables", expanded=False):

                code_rckd_var = None

                col1_remove_1, col2_remove_1 = st.columns([1.5,3.5])
                with col1_remove_1:

                    keep_or_delete_vars = st.selectbox("Select the option",['None','Rename','Copy','Keep','Delete'])

                    if keep_or_delete_vars == 'Delete':
                        #select_vars = []
                        select_or_insert_var_delete = st.selectbox("Select or insert variables", ['None','Select','Insert'])
                        if select_or_insert_var_delete == 'Select':
                            select_vars = st.multiselect("Select the variables to remove", options_columns)
                            code_rckd_var = \
f'''self.base.drop({select_vars},axis=1, inplace=True)'''
                        elif select_or_insert_var_delete == 'Insert':
                            select_vars = st.text_input("Insert the variables to remove")
                            select_vars = select_vars.split(',')
                            code_rckd_var = \
f'''self.base.drop({select_vars},axis=1, inplace=True)'''

                    elif keep_or_delete_vars == 'Keep':
                        select_or_insert_var_keep = st.selectbox("Select or insert variables", ['None','Select','Insert'])
                        if select_or_insert_var_keep == 'Select':
                            select_vars = st.multiselect("Select the variables to keep", options_columns)
                            code_rckd_var = \
f'''self.base = self.base[{select_vars}]'''
                        elif select_or_insert_var_keep == 'Insert':
                            select_vars = st.text_input("Insert the variables to keep")
                            select_vars = select_vars.split(',')
                            code_rckd_var = \
f'''self.base = self.base[{select_vars}]'''

                    elif keep_or_delete_vars == 'Rename':
                        new_name_var = st.text_input("Input the name of the new variables, Ex: col1:new_col1,col2:new_col2")
                        if new_name_var != '':
                            new_name_var = new_name_var.split(',')
                            new_name_var = [s.split(':') for s in new_name_var]
                            new_name_var = {s:x for s,x in new_name_var}
                        code_rckd_var = \
f'''self.base.rename(columns={new_name_var}, inplace=True)'''

                    elif keep_or_delete_vars == 'Copy':
                        new_name_var = st.text_input("Input the name of the new variable")
                        if new_name_var != '':
                            select_vars = st.selectbox("Select the variables to copy", options_columns)
                            code_rckd_var = \
f'''self.base['{new_name_var}'] = self.base['{select_vars}']'''


                if code_rckd_var:
                    st.code(code_rckd_var, language='python')

                submit_remov_var = st.button("Apply", key='apply_remov_var')
                if submit_remov_var:
                    if keep_or_delete_vars == 'Delete':
                        st.session_state[optionDataset].drop(select_vars,axis=1, inplace=True)
                        st.experimental_rerun()
                    elif keep_or_delete_vars == 'Keep':
                        st.session_state[optionDataset] = st.session_state[optionDataset][select_vars]
                        st.experimental_rerun()
                    elif keep_or_delete_vars == 'Rename':
                        st.session_state[optionDataset].rename(columns=new_name_var, inplace=True)
                        st.experimental_rerun()
                    elif keep_or_delete_vars == 'Copy':
                        st.session_state[optionDataset][new_name_var] = st.session_state[optionDataset][select_vars]
                        st.experimental_rerun()

            with st.expander("Split dataset", expanded=False):

                code_split_dataset = None

                col1_split_1, col2_split_1 = st.columns([2,3])
                with col1_split_1:
                    type_split = st.selectbox("Type of split", ['None','Simple split','Train and test data [train and test]','Train and test split [X_train, y_train, X_test, y_test]', 'Separation Column'])

                if type_split == 'Simple split':
                    st.info("Separate part of the dataset")
                    size_split = st.number_input("Input the size (%)", min_value=0.0, step=0.01, max_value=1.0)
                    save_in_new_variable_split = st.selectbox("Save split in new dataset?", ['No','Yes'])
                    if save_in_new_variable_split == 'Yes':
                        new_name_var_split = st.selectbox("Select the object to save the split", st.session_state['Objects'])
                    code_split_dataset = \
f'''# import the lib -> from sklearn.model_selection import train_test_split
reject_dataset, self.base = train_test_split(self.base, test_size={size_split}, random_state=42)'''

                    if code_split_dataset:
                        st.code(code_split_dataset, language='python')

                    submit_simple_split = st.button("Apply", key='submit_simple_split')
                    if submit_simple_split:
                        from sklearn.model_selection import train_test_split
                        if save_in_new_variable_split == 'No':
                            reject_dataset, st.session_state[optionDataset] = train_test_split(st.session_state[optionDataset], test_size=size_split, random_state=42)
                            st.write(f"Before the division: {reject_dataset.shape[0]+st.session_state[optionDataset].shape[0]} lines and {st.session_state[optionDataset].shape[1]} columns")
                            st.write(f"After the division: {st.session_state[optionDataset].shape[0]} lines and {st.session_state[optionDataset].shape[1]} columns")
                            del reject_dataset
                            st.success('Successfully!')
                            time.sleep(5)
                            st.experimental_rerun()
                        else:
                            if not new_name_var_split in st.session_state['dataset']:
                                reject_dataset, dataset = train_test_split(st.session_state[optionDataset], test_size=size_split, random_state=42)
                                st.session_state[new_name_var_split] = dataset
                                st.session_state['dataset'].append(new_name_var_split)
                                del reject_dataset
                                del dataset                                 
                                st.write(f"Before the division: {st.session_state[optionDataset].shape[0]} lines and {st.session_state[optionDataset].shape[1]} columns")
                                st.write(f"After the division: {st.session_state[new_name_var_split].shape[0]} lines and {st.session_state[new_name_var_split].shape[1]} columns")
                                st.success('Successfully!')
                                time.sleep(5)
                                st.experimental_rerun()
                            else:
                                st.error("This object is already in use, create a new object")

                elif type_split == 'Train and test data [train and test]':
                    st.info("Training and testing data separation, no label separation")
                    size_split_separeted = st.number_input("Input the size (%)", min_value=0.0, step=0.01, max_value=1.0)
                    
                    submit_separeted_train_test = st.button("Apply", key='submit_simple_split')
                    if submit_separeted_train_test:
                        from sklearn.model_selection import train_test_split
                        st.session_state[optionDataset+'_train'], st.session_state[optionDataset+'_test'] = train_test_split(st.session_state[optionDataset], test_size=size_split_separeted, random_state=42)
                        st.session_state['dataset'].append(optionDataset+'_train')
                        st.session_state['dataset'].append(optionDataset+'_test')
                        st.success('Successfully!')
                        time.sleep(5)
                        st.experimental_rerun()

                elif type_split == 'Train and test split [X_train, y_train, X_test, y_test]':
                    st.info("Separation of training and test data with label separation")
                    var_target_split = st.selectbox("Input the target variable", options_columns)

                    size_test_split = st.number_input("Input the test size (%)", min_value=0.0, step=0.01, max_value=1.0)

                    code_split_dataset = \
f'''# import the lib -> from sklearn.model_selection import train_test_split
X = self.base.drop(['{var_target_split}'],axis=1)
y = self.base[['{var_target_split}']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={size_test_split}, random_state=42)'''

                    if code_split_dataset:
                        st.code(code_split_dataset, language='python')

                    submit_simple_split = st.button("Apply", key='submit_simple_split')
                    if submit_simple_split:
                        from sklearn.model_selection import train_test_split

                        df_X = st.session_state[optionDataset].drop(var_target_split,axis=1)
                        df_y = st.session_state[optionDataset][[var_target_split]]

                        st.session_state[optionDataset+'_X_train'], st.session_state[optionDataset+'_X_test'], st.session_state[optionDataset+'_y_train'], st.session_state[optionDataset+'_y_test'] = train_test_split(df_X, df_y, test_size=size_test_split, random_state=42)
                        st.session_state['dataset'].append(optionDataset+'_X_train')
                        st.session_state['dataset'].append(optionDataset+'_X_test')
                        st.session_state['dataset'].append(optionDataset+'_y_train')
                        st.session_state['dataset'].append(optionDataset+'_y_test')
                        del df_X
                        del df_y
                        st.write(f"{optionDataset} X train: {st.session_state[optionDataset+'_X_train'].shape[0]} lines and {st.session_state[optionDataset+'_X_train'].shape[1]} columns")
                        st.write(f"{optionDataset} y train: {st.session_state[optionDataset+'_y_train'].shape[0]} lines and {st.session_state[optionDataset+'_y_train'].shape[1]} columns")
                        st.write(f"{optionDataset} X test: {st.session_state[optionDataset+'_X_test'].shape[0]} lines and {st.session_state[optionDataset+'_X_test'].shape[1]} columns")
                        st.write(f"{optionDataset} y test: {st.session_state[optionDataset+'_y_test'].shape[0]} lines and {st.session_state[optionDataset+'_y_test'].shape[1]} columns")
                        st.success('Successfully!')

                elif type_split == 'Separation Column':
                    st.info("Variable separation")
                    var_target_separation = st.selectbox("Input the target variable", options_columns)
                    objs_to_save_separation = [obj for obj in st.session_state['Objects'] if obj not in st.session_state['dataset']]
                    select_object_to_separation_save = st.selectbox("Select the object to save", objs_to_save_separation)

                    if var_target_separation != 'None' and select_object_to_separation_save:

                        submit_label_separation = st.button("Apply", key='submit_label_separation')
                        if submit_label_separation:

                            st.session_state[select_object_to_separation_save] = st.session_state[optionDataset][[var_target_separation]]
                            st.session_state['dataset'].append(select_object_to_separation_save)

                            #st.session_state[optionDataset+'_X'] = st.session_state[optionDataset].drop(var_target_separation,axis=1)
                            #st.session_state['dataset'].append(optionDataset+'_X')
                            #st.session_state[optionDataset+'_y'] = st.session_state[optionDataset][[var_target_separation]]
                            #st.session_state['dataset'].append(optionDataset+'_y')
                            st.success('Successfully!')
                            time.sleep(1.5)
                            st.experimental_rerun()

            with st.expander("Filter dataset", expanded=False):

                col1_filter_manys, col2_filter_manys, col3_filter_manys, col4_filter_manys, col5_filter_manys  = st.columns([0.8,0.4,0.8,0.8,2])
                with col1_filter_manys:
                    column_to_filter = st.selectbox("Select the column", st.session_state[optionDataset].columns)
                with col2_filter_manys:
                    comparation = st.selectbox("Conditional", ['>','>=','<','<=','==','!='])
                with col3_filter_manys:
                    type_value_to_filter = st.selectbox("Type filter", ["Numerical", "Categorical"])
                with col4_filter_manys:
                    if type_value_to_filter == 'Categorical':
                        value_to_filter = st.text_input("Filter by")
                    elif type_value_to_filter == 'Numerical':
                        value_to_filter = st.number_input("Filter by", value=1234567890)

                if type_value_to_filter == 'Categorical':
                    code_filter = f'''self.base = self.base[self.base["{column_to_filter}"] {comparation} "{value_to_filter}"]'''
                    st.code(code_filter, language='python')
                elif type_value_to_filter == 'Numerical':
                    code_filter = f'''self.base = self.base[self.base["{column_to_filter}"] {comparation} {value_to_filter}]'''
                    st.code(code_filter, language='python') 
                
                submit_filter = st.button("Apply", key='apply_filter')
                if submit_filter:
                    
                    if comparation == '>':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] > value_to_filter]
                    elif comparation == '>=':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] >= value_to_filter]
                    elif comparation == '<':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] < value_to_filter]
                    elif comparation == '<=':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] <= value_to_filter]
                    elif comparation == '==':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] == value_to_filter]
                    elif comparation == '!=':
                        st.session_state[optionDataset] = st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] != value_to_filter]
                    st.experimental_rerun()

            with st.expander("Reset index", expanded=False):

                submit_reset_index = st.button("Apply", key='submit_reset_index')

                code_reset_index = \
f'''self.base.reset_index(drop=True, inplace=True)'''

                st.code(code_reset_index, language='python')
                if submit_reset_index:
                    st.session_state[optionDataset].reset_index(drop=True, inplace=True)
                    time.sleep(1)
                    st.experimental_rerun()

            with st.expander("Remove duplicates", expanded=False):

                col1_dropduplicate, col2_dropduplicate, col3_dropduplicate  = st.columns([1,2,3])
                with col1_dropduplicate:
                    dropduplicate_select_or_all = st.selectbox("Drop all or select variables?",['None','All','Select'])
                with col2_dropduplicate:
                    if dropduplicate_select_or_all == 'Select':
                        column_to_dropduplicate = st.multiselect("Select the columns", st.session_state[optionDataset].columns)

                if dropduplicate_select_or_all != 'None':
                    if dropduplicate_select_or_all == 'Select':
                        code_dropduplicate = \
    f'''self.base.drop_duplicates(subset={column_to_dropduplicate}, inplace=True)'''
                        if column_to_dropduplicate:
                            st.code(code_dropduplicate, language='python')
                    else:
                        code_dropduplicate = \
    f'''self.base.drop_duplicates(inplace=True)'''
                        st.code(code_dropduplicate, language='python')

                submit_dropduplicate = st.button("Apply", key='apply_dropduplicate')
                if submit_dropduplicate:
                    if dropduplicate_select_or_all == 'Select':
                        st.session_state[optionDataset].drop_duplicates(subset=column_to_dropduplicate, inplace=True)
                        st.experimental_rerun()
                    else:
                        st.session_state[optionDataset].drop_duplicates(inplace=True)
                        st.experimental_rerun()
                
            with st.expander("Data imputation", expanded=False):

                code_vars_imputation = None

                col1_imputation_1, col2_imputation_1 = st.columns([1.5,3.5])
                with col1_imputation_1:
                    number_or_object_var_imputation = st.selectbox("Categorical or numeric variables?",['None','Category','Numerical'])
                
                col1_imputation_2, col2_imputation_2 = st.columns([1.5,3.5])
                with col1_imputation_2:
                    if number_or_object_var_imputation != 'None':
                        if number_or_object_var_imputation == 'Category':
                            imputation = st.selectbox("Type of imputation",['None','Most frequent'])
                        elif number_or_object_var_imputation == 'Numerical':
                            imputation = st.selectbox("Type of imputation",['None','Mean','Median'])

                        if imputation != 'None':
                            select_or_all_vars_imputation = st.selectbox("Imputation all or select variables?",['None','All','Select'])
                        
                            if select_or_all_vars_imputation == 'All':
                                if number_or_object_var_imputation == 'Category':
                                    vars_imputation = [col for col in st.session_state[optionDataset].select_dtypes(include=[np.object,'category']).columns]
                                    if imputation == 'Most frequent':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].value_counts().idxmax(), inplace=True)'''

                                    elif imputation == 'Mean':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].mean(), inplace=True)'''

                                    elif imputation == 'Median':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].median(), inplace=True)'''

                                    
                                elif number_or_object_var_imputation == 'Numerical':
                                    vars_imputation = [col for col in st.session_state[optionDataset].select_dtypes(include=[np.number]).columns]
                                    if imputation == 'Most frequent':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].value_counts().idxmax(), inplace=True)'''

                                    elif imputation == 'Mean':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].mean(), inplace=True)'''

                                    elif imputation == 'Median':
                                        code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].median(), inplace=True)'''

                            elif select_or_all_vars_imputation == 'Select':
                                if number_or_object_var_imputation == 'Category':
                                    vars_imputation = st.multiselect("Select the columns",st.session_state[optionDataset].select_dtypes(include=[np.object,'category']).columns)
                                    if vars_imputation:
                                        if imputation == 'Most frequent':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].value_counts().idxmax(), inplace=True)'''

                                        elif imputation == 'Mean':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].mean(), inplace=True)'''

                                        elif imputation == 'Median':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].median(), inplace=True)'''
                                
                                elif number_or_object_var_imputation == 'Numerical':
                                    vars_imputation = st.multiselect("Select the columns",st.session_state[optionDataset].select_dtypes(include=[np.number]).columns)
                                    if vars_imputation:
                                        if imputation == 'Most frequent':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].value_counts().idxmax(), inplace=True)'''

                                        elif imputation == 'Mean':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].mean(), inplace=True)'''

                                        elif imputation == 'Median':
                                            code_vars_imputation = \
f'''for col in {vars_imputation}:
    self.base[col].fillna(self.base[col].median(), inplace=True)'''

                if code_vars_imputation != None:
                    st.code(code_vars_imputation, language='python')

                submit_imputation = st.button("Apply", key='apply_imputation')
                if submit_imputation:
                    if imputation == 'Most frequent':
                        for col in vars_imputation:
                            st.session_state[optionDataset][col].fillna(st.session_state[optionDataset][col].value_counts().idxmax(), inplace=True)
                    elif imputation == 'Mean':
                        for col in vars_imputation:
                            st.session_state[optionDataset][col].fillna(st.session_state[optionDataset][col].mean(), inplace=True)
                    elif imputation == 'Median':
                        for col in vars_imputation:
                            st.session_state[optionDataset][col].fillna(st.session_state[optionDataset][col].median(), inplace=True)

            with st.expander("Replace values", expanded=False):

                code_to_replace = None
                new_values_to_replace = None
                old_values_to_replace = None

                col1_replace_1, col2_replace_1 = st.columns([1.5,3.5])
                with col1_replace_1:

                    columns_to_replace = st.selectbox("Select the column to replace", options_columns)
                    
                    if columns_to_replace != 'None':
                        string_or_number_to_replace = st.selectbox("Data type to replace?",['None','String','Integer','Float','NaN'])

                        if string_or_number_to_replace == 'String':
                            old_values_to_replace = st.text_input("Input the value to replace")
                            new_values_to_replace = st.text_input("Input the new value")
                            code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: "{new_values_to_replace}" if x == "{old_values_to_replace}" else x)'''
                        if string_or_number_to_replace == 'Integer':
                            old_values_to_replace = st.number_input("Input the value to replace", step=1)
                            new_values_to_replace = st.number_input("Input the new value", step=1)
                            code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: {new_values_to_replace} if x == {old_values_to_replace} else x)'''
                        if string_or_number_to_replace == 'Float':
                            old_values_to_replace = st.number_input("Input the value to replace", step=0.01)
                            new_values_to_replace = st.number_input("Input the new value", step=0.01)
                            code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: {new_values_to_replace} if x == {old_values_to_replace} else x)'''
                        if string_or_number_to_replace == 'NaN':
                            string_or_number_to_replace_nan = st.selectbox("Data type to replace empty value?",['None','String','Integer','Float'])
                            if string_or_number_to_replace_nan == 'String':
                                new_values_to_replace = st.text_input("Input the new value")
                                code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: "{new_values_to_replace}" if x is np.nan else x)'''
                            elif string_or_number_to_replace_nan == 'Integer':
                                new_values_to_replace = st.number_input("Input the new value", step=1)
                                code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: {new_values_to_replace} if x is np.nan else x)'''
                            elif string_or_number_to_replace_nan == 'Float':
                                new_values_to_replace = st.number_input("Input the new value", step=0.01)
                                code_to_replace = \
f'''self.base["{columns_to_replace}"] = self.base["{columns_to_replace}"].apply(lambda x: {new_values_to_replace} if x is np.nan else x)'''

                if code_to_replace:
                    st.code(code_to_replace, language='python')

                submit_replace = st.button("Apply", key='apply_replace')
                if submit_replace:
                    if string_or_number_to_replace == 'NaN':
                        st.session_state[optionDataset][columns_to_replace] = st.session_state[optionDataset][columns_to_replace].apply(lambda x: new_values_to_replace if x is np.nan else x)
                        st.experimental_rerun()
                    else:
                        st.session_state[optionDataset][columns_to_replace] = st.session_state[optionDataset][columns_to_replace].apply(lambda x: new_values_to_replace if x == old_values_to_replace else x)
                        st.experimental_rerun()

            with st.expander("Binning", expanded=False):

                code_to_binning = None

                col1_binning_1, col2_binning_1 = st.columns([1.5,3.5])
                with col1_binning_1:
                    columns_to_binning = st.selectbox("Select the column to binnig", options_columns)

                    if columns_to_binning != 'None':

                        auto_bin_or_range = st.selectbox("Auto bin or input breaks", ['None','Auto bin', 'Input breaks'])

                        if auto_bin_or_range == 'Auto bin':
                            n_auto_bin = st.number_input("Input number of bins", step=1)
                            code_to_binning = \
f'''self.base['{columns_to_binning}_2'] = pd.cut(self.base['{columns_to_binning}'], bins={n_auto_bin}, include_lowest=True)
self.base['{columns_to_binning}_group'] = self.base['{columns_to_binning}_2'].apply(lambda x: str(x.left) +'-'+ str(x.right))
del self.base['{columns_to_binning}_2']'''
                            
                            if n_auto_bin > 0:
                                submit_binning = st.button("Apply", key='apply_binning_1')
                                if submit_binning:
                                    st.session_state[optionDataset][columns_to_binning+'_2'] = pd.cut(st.session_state[optionDataset][columns_to_binning], bins=n_auto_bin, include_lowest=True)
                                    st.session_state[optionDataset][columns_to_binning+'_group'] = st.session_state[optionDataset][columns_to_binning+'_2'].apply(lambda x: str(x.left) +'-'+ str(x.right))
                                    del st.session_state[optionDataset][columns_to_binning+'_2']

                        elif auto_bin_or_range == 'Input breaks':
                            range_to_binning = st.text_input("Input the breaks, ex: 17,25,40,60,120")
                            if range_to_binning != '':
                                list_range_to_binning = [int(num) for num in range_to_binning.split(',')]
                                labels_to_binning = st.text_input("Input the labels, ex: '18-25', '26-40', '41-60', '61-'")
                                list_labels_to_binning = labels_to_binning.split(',')
                                if labels_to_binning != '':
                                    code_to_binning = \
f'''self.base["{columns_to_binning}_group"] = pd.cut(self.base["{columns_to_binning}"], {list_range_to_binning}, labels={list_labels_to_binning}, include_lowest=True)'''
                                    submit_binning = st.button("Apply", key='apply_binning_2')
                                    if submit_binning:
                                        st.session_state[optionDataset][columns_to_binning+'_group'] = pd.cut(st.session_state[optionDataset][columns_to_binning], list_range_to_binning, labels=list_labels_to_binning, include_lowest=True)
                if code_to_binning:
                    st.code(code_to_binning, language='python')

            with st.expander("LabelEncoder", expanded=False):

                code_labelencoder = None
                
                col1_labelencoder_1, col2_labelencoder_1 = st.columns([1.5,3.5])
                with col1_labelencoder_1:
                    insert_or_select_columns = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_labelencoder')

                    if insert_or_select_columns == 'Insert':
                        insert_columns_to_labelencoder = st.text_input("Insert the columns to apply labelencoder")
                        columns_to_labelencoder = insert_columns_to_labelencoder.split(',')
                        if insert_columns_to_labelencoder:
                            code_labelencoder = \
f'''# import the lib -> from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for col in {columns_to_labelencoder}:
    self.base[col] = label_encoder.fit_transform(self.base[col])'''
                        
                    elif insert_or_select_columns == 'Select':
                        columns_to_labelencoder = st.multiselect("Select the columns to apply labelencoder", options_columns)
                        if columns_to_labelencoder:
                            code_labelencoder = \
f'''# import the lib -> from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for col in {columns_to_labelencoder}:
    self.base[col] = label_encoder.fit_transform(self.base[col])'''

                if code_labelencoder:
                    st.code(code_labelencoder, language='python')      
                            
                submit_labelencoder = st.button("Apply", key='apply_labelencoder')
                if submit_labelencoder:
                    if columns_to_labelencoder:

                        label_encoder = preprocessing.LabelEncoder()
                        for col in columns_to_labelencoder:
                            st.session_state[optionDataset][col] = label_encoder.fit_transform(st.session_state[optionDataset][col])
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.warning("Input the columns")
                        
            with st.expander("OneHotEncoder", expanded=False):

                code_onehotencoder = None
                
                col1_onehotencoder_1, col2_onehotencoder_1 = st.columns([1.5,3.5])
                with col1_onehotencoder_1:
                    insert_or_select_columns = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_onehotencoder')

                    if insert_or_select_columns == 'Insert':
                        insert_columns_to_onehotencoder = st.text_input("Insert the columns to apply OneHotEncoder")
                        columns_to_onehotencoder = insert_columns_to_onehotencoder.split(',')
                        st.write(columns_to_onehotencoder)
                        keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'])
                        if insert_columns_to_onehotencoder:
                            if keep_or_delete_columns == 'Yes':
                                code_onehotencoder = \
f'''# import the lib -> import pandas as pd
dummies = pd.get_dummies(self.base[{columns_to_onehotencoder}], drop_first=False)
self.base = pd.concat([self.base.drop({columns_to_onehotencoder},axis=1), dummies], axis=1)'''
                            else:
                                code_onehotencoder = \
f'''# import the lib -> import pandas as pd
dummies = pd.get_dummies(self.base[{columns_to_onehotencoder}], drop_first=False)
self.base = pd.concat([self.base, dummies], axis=1)'''

                    elif insert_or_select_columns == 'Select':
                        columns_to_onehotencoder = st.multiselect("Select the columns to apply OneHotEncoder", options_columns)
                        keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'])
                        if columns_to_onehotencoder:
                            if keep_or_delete_columns == 'Yes':
                                code_onehotencoder = \
f'''# import the lib -> import pandas as pd
dummies = pd.get_dummies(self.base[{columns_to_onehotencoder}], drop_first=False)
self.base = pd.concat([self.base.drop({columns_to_onehotencoder},axis=1), dummies], axis=1)'''
                            else:
                                code_onehotencoder = \
f'''# import the lib -> import pandas as pd
dummies = pd.get_dummies(self.base[{columns_to_onehotencoder}], drop_first=False)
self.base = pd.concat([self.base, dummies], axis=1)'''
                
                if code_onehotencoder:
                    st.code(code_onehotencoder, language='python')

                submit_onehotencoder = st.button("Apply", key='apply_onehotencoder')
                if submit_onehotencoder:
                    if columns_to_onehotencoder:
                        if keep_or_delete_columns == 'Yes':
                            dummies = pd.get_dummies(st.session_state[optionDataset][columns_to_onehotencoder], drop_first=False)
                            st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset].drop(columns_to_onehotencoder,axis=1), dummies], axis=1)
                            time.sleep(1)
                            st.experimental_rerun()
                        else:
                            dummies = pd.get_dummies(st.session_state[optionDataset][columns_to_onehotencoder], drop_first=False)
                            st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset], dummies], axis=1)
                            time.sleep(1)
                            st.experimental_rerun()
                    else:
                        st.warning("Input the columns")
            
            with st.expander("StandardScaler", expanded=False):

                StandardScalerType = option_menu('', ['Create scaler','Apply transform'], 
                    default_index=0, orientation="horizontal",
                    styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                            "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
                )

                if StandardScalerType == 'Create scaler':

                    code_standardscaler = None
                    
                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        insert_or_select_columns = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_standardscaler')
                    
                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        save_object_standardscaler = st.checkbox("Save object")

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if save_object_standardscaler:
                            name_object_standardscaler = st.text_input("Input object name to save")

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if insert_or_select_columns == 'Insert':
                            insert_columns_to_standardscaler = st.text_input("Insert the columns to apply StandardScaler")
                            columns_to_standardscaler = insert_columns_to_standardscaler.split(',')
                            keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_standardscaler')
                            if insert_columns_to_standardscaler:
                                if keep_or_delete_columns == 'Yes':
                                    if save_object_standardscaler and name_object_standardscaler != '':
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler},axis=1), df_std], axis=1)
pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))'''
                                    else:
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler},axis=1), df_std], axis=1)'''
                                else:
                                    if save_object_standardscaler and name_object_standardscaler != '':
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)
pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))'''
                                    else:
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)'''

                        elif insert_or_select_columns == 'Select':
                            columns_to_standardscaler = st.multiselect("Select the columns to apply StandardScaler", options_columns)
                            keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_standardscaler')
                            if columns_to_standardscaler:
                                if keep_or_delete_columns == 'Yes':
                                    if save_object_standardscaler and name_object_standardscaler != '':
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler},axis=1), df_std], axis=1)
pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))'''
                                    else:
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler},axis=1), df_std], axis=1)'''
                                else:
                                    if save_object_standardscaler and name_object_standardscaler != '':
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)
pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))'''
                                    else:
                                        code_standardscaler = \
f'''# import the lib -> from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_std = scaler.fit(self.base[{columns_to_standardscaler}]).transform(self.base[{columns_to_standardscaler}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)'''

                    if code_standardscaler:
                        st.code(code_standardscaler, language='python')

                    submit_standardscaler = st.button("Apply", key='apply_standardscaler')
                    if submit_standardscaler:
                        if columns_to_standardscaler:
                            if keep_or_delete_columns == 'Yes':
                                if save_object_standardscaler and name_object_standardscaler != '':
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit(st.session_state[optionDataset][columns_to_standardscaler]).transform(st.session_state[optionDataset][columns_to_standardscaler])
                                    name_col_std = [col + '_std' for col in st.session_state[optionDataset][columns_to_standardscaler].columns]
                                    df_std = pd.DataFrame(df_std, columns=name_col_std)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset].drop(columns_to_standardscaler,axis=1), df_std], axis=1)
                                    pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit(st.session_state[optionDataset][columns_to_standardscaler]).transform(st.session_state[optionDataset][columns_to_standardscaler])
                                    name_col_std = [col + '_std' for col in st.session_state[optionDataset][columns_to_standardscaler].columns]
                                    df_std = pd.DataFrame(df_std, columns=name_col_std)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset].drop(columns_to_standardscaler,axis=1), df_std], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                            else:
                                if save_object_standardscaler and name_object_standardscaler != '':
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit(st.session_state[optionDataset][columns_to_standardscaler]).transform(st.session_state[optionDataset][columns_to_standardscaler])
                                    name_col_std = [col + '_std' for col in st.session_state[optionDataset][columns_to_standardscaler].columns]
                                    df_std = pd.DataFrame(df_std, columns=name_col_std)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset], df_std], axis=1)
                                    pickle.dump(scaler, open(f'saved_models/{name_object_standardscaler}.pkl','wb'))
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit(st.session_state[optionDataset][columns_to_standardscaler]).transform(st.session_state[optionDataset][columns_to_standardscaler])
                                    name_col_std = [col + '_std' for col in st.session_state[optionDataset][columns_to_standardscaler].columns]
                                    df_std = pd.DataFrame(df_std, columns=name_col_std)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset], df_std], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                        else:
                            st.warning("Input the columns")

                elif StandardScalerType == 'Apply transform':

                    sc = None
                    select_data_to_apply_sc = None
                    columns_to_standardscaler_sc = None
                    insert_or_select_columns_sc = None
                    code_standardscaler_sc = None

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        options_files = st.selectbox('Select object', list_files)
                        if options_files != 'None':
                            sc = pickle.load(open(f'saved_models/{options_files}','rb'))

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if sc:
                            select_data_to_apply_sc = st.selectbox("Select dataset", st.session_state['dataset'])

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if select_data_to_apply_sc:
                            insert_or_select_columns_sc = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_sc')

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if insert_or_select_columns_sc == 'Insert':
                            insert_columns_to_standardscaler_sc = st.text_input("Insert the columns to apply transform")
                            columns_to_standardscaler_sc = insert_columns_to_standardscaler_sc.split(',')
                            keep_or_delete_columns_sc = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_sc')
                            if insert_columns_to_standardscaler_sc:
                                if keep_or_delete_columns_sc == 'Yes':
                                    code_standardscaler_sc = \
f'''# import the lib -> import pickle
sc = pickle.load(open(f'saved_models/{options_files}','rb'))
df_std = sc.transform(self.base[{columns_to_standardscaler_sc}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler_sc}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler_sc},axis=1), df_std], axis=1)'''
                                else:
                                    code_standardscaler_sc = \
f'''# import the lib -> import pickle
sc = pickle.load(open(f'saved_models/{options_files}','rb'))
df_std = sc.transform(self.base[{columns_to_standardscaler_sc}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler_sc}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)'''
                        elif insert_or_select_columns_sc == 'Select':
                            columns_to_standardscaler_sc = st.multiselect("Select the columns to apply transform", st.session_state[select_data_to_apply_sc].columns)
                            keep_or_delete_columns_sc = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_sc')
                            if columns_to_standardscaler_sc:
                                if keep_or_delete_columns_sc == 'Yes':
                                    code_standardscaler_sc = \
f'''# import the lib -> import pickle
sc = pickle.load(open(f'saved_models/{options_files}','rb'))
df_std = sc.transform(self.base[{columns_to_standardscaler_sc}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler_sc}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base.drop({columns_to_standardscaler_sc},axis=1), df_std], axis=1)'''
                                else:
                                    code_standardscaler_sc = \
f'''# import the lib -> import pickle
sc = pickle.load(open(f'saved_models/{options_files}','rb'))
df_std = sc.transform(self.base[{columns_to_standardscaler_sc}])
name_col_std = [col + '_std' for col in self.base[{columns_to_standardscaler_sc}].columns]
df_std = pd.DataFrame(df_std, columns=name_col_std)
self.base = pd.concat([self.base, df_std], axis=1)'''

                    if code_standardscaler_sc:
                        st.code(code_standardscaler_sc, language='python')

                    if sc and select_data_to_apply_sc and columns_to_standardscaler_sc:
                        button_apply_sc = st.button("Apply", key='button_apply_sc')
                        if button_apply_sc:
                            if keep_or_delete_columns_sc == 'Yes':
                                #sc = pickle.load(open(f'saved_models/{options_files}','rb'))
                                df_std = sc.transform(st.session_state[select_data_to_apply_sc][columns_to_standardscaler_sc])
                                name_col_std = [col + '_std' for col in st.session_state[select_data_to_apply_sc][columns_to_standardscaler_sc].columns]
                                df_std = pd.DataFrame(df_std, columns=name_col_std)
                                st.session_state[select_data_to_apply_sc] = pd.concat([st.session_state[select_data_to_apply_sc].drop(columns_to_standardscaler_sc,axis=1), df_std], axis=1)
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                #sc = pickle.load(open(f'saved_models/{options_files}','rb'))
                                df_std = sc.transform(st.session_state[select_data_to_apply_sc][columns_to_standardscaler_sc])
                                name_col_std = [col + '_std' for col in st.session_state[select_data_to_apply_sc][columns_to_standardscaler_sc].columns]
                                df_std = pd.DataFrame(df_std, columns=name_col_std)
                                st.session_state[select_data_to_apply_sc] = pd.concat([st.session_state[select_data_to_apply_sc], df_std], axis=1)
                                time.sleep(1)
                                st.experimental_rerun()
                    
            with st.expander("MinMaxScaler", expanded=False):

                MinMaxScalerType = option_menu('', ['Create scaler','Apply transform'], 
                    default_index=0, orientation="horizontal",
                    styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                            "nav-link": {"font-size": "13px","--hover-color": "#eee"}}, key='sub_menu_minMaxScaler'
                )
                
                if MinMaxScalerType == 'Create scaler':
                    #minmaxscaler
                    code_minmaxscaler = None
                    
                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        insert_or_select_columns = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_minmaxscaler')

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        save_object_minmaxscaler = st.checkbox("Save object", key='save_object_minmaxscaler')

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if save_object_minmaxscaler:
                            name_object_minmaxscaler = st.text_input("Input object name to save")

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if insert_or_select_columns == 'Insert':
                            insert_columns_to_minmaxscaler = st.text_input("Insert the columns to apply MinMaxScaler")
                            columns_to_minmaxscaler = insert_columns_to_minmaxscaler.split(',')
                            keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_minmaxscaler')
                            if insert_columns_to_minmaxscaler:
                                if keep_or_delete_columns == 'Yes':
                                    if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minmaxscaler},axis=1), df_mms], axis=1)
pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))'''
                                    else:
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minmaxscaler},axis=1), df_mms], axis=1)'''
                                else:
                                    if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)
pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))'''
                                    else:
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)'''

                        elif insert_or_select_columns == 'Select':
                            columns_to_minmaxscaler = st.multiselect("Select the columns to apply MinMaxScaler", options_columns)
                            keep_or_delete_columns = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_minmaxscaler')
                            if columns_to_minmaxscaler:
                                if keep_or_delete_columns == 'Yes':
                                    if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minmaxscaler},axis=1), df_mms], axis=1)
pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))'''
                                    else:
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minmaxscaler},axis=1), df_mms], axis=1)'''
                                else:
                                    if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)
pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))'''
                                    else:
                                        code_minmaxscaler = \
f'''# import the lib -> from sklearn import preprocessing
minMaxScaler = preprocessing.MinMaxScaler()
df_mms = minMaxScaler.fit(self.base[{columns_to_minmaxscaler}]).transform(self.base[{columns_to_minmaxscaler}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minmaxscaler}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)'''

                    if code_minmaxscaler:
                        st.code(code_minmaxscaler, language='python')

                    submit_minmaxscaler = st.button("Apply", key='apply_minmaxscaler')
                    if submit_minmaxscaler:
                        if columns_to_minmaxscaler:
                            if keep_or_delete_columns == 'Yes':
                                if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                    minMaxScaler = preprocessing.MinMaxScaler()
                                    df_mms = minMaxScaler.fit(st.session_state[optionDataset][columns_to_minmaxscaler]).transform(st.session_state[optionDataset][columns_to_minmaxscaler])
                                    name_col_mms = [col + '_mms' for col in st.session_state[optionDataset][columns_to_minmaxscaler].columns]
                                    df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset].drop(columns_to_minmaxscaler,axis=1), df_mms], axis=1)
                                    pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    minMaxScaler = preprocessing.MinMaxScaler()
                                    df_mms = minMaxScaler.fit(st.session_state[optionDataset][columns_to_minmaxscaler]).transform(st.session_state[optionDataset][columns_to_minmaxscaler])
                                    name_col_mms = [col + '_mms' for col in st.session_state[optionDataset][columns_to_minmaxscaler].columns]
                                    df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset].drop(columns_to_minmaxscaler,axis=1), df_mms], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                            else:
                                if save_object_minmaxscaler and name_object_minmaxscaler != '':
                                    minMaxScaler = preprocessing.MinMaxScaler()
                                    df_mms = minMaxScaler.fit(st.session_state[optionDataset][columns_to_minmaxscaler]).transform(st.session_state[optionDataset][columns_to_minmaxscaler])
                                    name_col_mms = [col + '_mms' for col in st.session_state[optionDataset][columns_to_minmaxscaler].columns]
                                    df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset], df_mms], axis=1)
                                    pickle.dump(minMaxScaler, open(f'saved_models/{name_object_minmaxscaler}.pkl','wb'))
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    minMaxScaler = preprocessing.MinMaxScaler()
                                    df_mms = minMaxScaler.fit(st.session_state[optionDataset][columns_to_minmaxscaler]).transform(st.session_state[optionDataset][columns_to_minmaxscaler])
                                    name_col_mms = [col + '_mms' for col in st.session_state[optionDataset][columns_to_minmaxscaler].columns]
                                    df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                    st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset], df_mms], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                        else:
                            st.warning("Input the columns")

                elif MinMaxScalerType == 'Apply transform':
                    
                    mms = None
                    select_data_to_apply_mms = None
                    columns_to_minMaxScaler_mms = None
                    insert_or_select_columns_mms = None
                    code_minMaxScaler_mms = None

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        options_files = st.selectbox('Select object', list_files)
                        if options_files != 'None':
                            mms = pickle.load(open(f'saved_models/{options_files}','rb'))

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if mms:
                            select_data_to_apply_mms = st.selectbox("Select dataset", st.session_state['dataset'])

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if select_data_to_apply_mms:
                            insert_or_select_columns_mms = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_mms')

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if insert_or_select_columns_mms == 'Insert':
                            insert_columns_to_minMaxScaler_mms = st.text_input("Insert the columns to apply transform")
                            columns_to_minMaxScaler_mms = insert_columns_to_minMaxScaler_mms.split(',')
                            keep_or_delete_columns_mms = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_mms')
                            if insert_columns_to_minMaxScaler_mms:
                                if keep_or_delete_columns_mms == 'Yes':
                                    code_minMaxScaler_mms = \
f'''# import the lib -> import pickle
mms = pickle.load(open(f'saved_models/{options_files}','rb'))
df_mms = mms.transform(self.base[{columns_to_minMaxScaler_mms}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minMaxScaler_mms}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minMaxScaler_mms},axis=1), df_mms], axis=1)'''
                                else:
                                    code_minMaxScaler_mms = \
f'''# import the lib -> import pickle
mms = pickle.load(open(f'saved_models/{options_files}','rb'))
df_mms = mms.transform(self.base[{columns_to_minMaxScaler_mms}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minMaxScaler_mms}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)'''
                        elif insert_or_select_columns_mms == 'Select':
                            columns_to_minMaxScaler_mms = st.multiselect("Select the columns to apply transform", st.session_state[select_data_to_apply_mms].columns)
                            keep_or_delete_columns_mms = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_mms')
                            if columns_to_minMaxScaler_mms:
                                if keep_or_delete_columns_mms == 'Yes':
                                    code_minMaxScaler_mms = \
f'''# import the lib -> import pickle
mms = pickle.load(open(f'saved_models/{options_files}','rb'))
df_mms = mms.transform(self.base[{columns_to_minMaxScaler_mms}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minMaxScaler_mms}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base.drop({columns_to_minMaxScaler_mms},axis=1), df_mms], axis=1)'''
                                else:
                                    code_minMaxScaler_mms = \
f'''# import the lib -> import pickle
mms = pickle.load(open(f'saved_models/{options_files}','rb'))
df_mms = mms.transform(self.base[{columns_to_minMaxScaler_mms}])
name_col_mms = [col + '_mms' for col in self.base[{columns_to_minMaxScaler_mms}].columns]
df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
self.base = pd.concat([self.base, df_mms], axis=1)'''

                    if code_minMaxScaler_mms:
                        st.code(code_minMaxScaler_mms, language='python')

                    if mms and select_data_to_apply_mms and columns_to_minMaxScaler_mms:
                        button_apply_mms = st.button("Apply", key='button_apply_mms')
                        if button_apply_mms:
                            if keep_or_delete_columns_mms == 'Yes':
                                #mms = pickle.load(open(f'saved_models/{options_files}','rb'))
                                df_mms = mms.transform(st.session_state[select_data_to_apply_mms][columns_to_minMaxScaler_mms])
                                name_col_mms = [col + '_mms' for col in st.session_state[select_data_to_apply_mms][columns_to_minMaxScaler_mms].columns]
                                df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                st.session_state[select_data_to_apply_mms] = pd.concat([st.session_state[select_data_to_apply_mms].drop(columns_to_minMaxScaler_mms,axis=1), df_mms], axis=1)
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                #mms = pickle.load(open(f'saved_models/{options_files}','rb'))
                                df_mms = mms.transform(st.session_state[select_data_to_apply_mms][columns_to_minMaxScaler_mms])
                                name_col_mms = [col + '_mms' for col in st.session_state[select_data_to_apply_mms][columns_to_minMaxScaler_mms].columns]
                                df_mms = pd.DataFrame(df_mms, columns=name_col_mms)
                                st.session_state[select_data_to_apply_mms] = pd.concat([st.session_state[select_data_to_apply_mms], df_mms], axis=1)
                                time.sleep(1)
                                st.experimental_rerun()

            with st.expander("Outlier Detection", expanded=False):

                st.image(Image.open('images/outliers.png'), width=700)

                st.warning("One way to detect outliers is through the interquartile range, which is the difference between the third and first quartiles. A commonly used rule is that an outlier is in the range less than 1.5 x interquartile range of the first quartile or greater than 1.5 x interquartile range of the third quartile.")

                st.subheader("Calculation")
                st.write("Inferior limit = First Quartile - 1.5 * (Third Quartile - First Quartile)")
                st.write("Over limit = Third Quartile + 1,5 * (Third Quartile - First Quartile)")
                
                var_graph_outlier  = st.multiselect("Select the variables", st.session_state[optionDataset].select_dtypes(include=[np.number]).columns, key ='var_graph_outlier')

                show_graphs = st.checkbox("Show graphs")
                if show_graphs:
                    import libs.EDA_graphs as EDA
                    eda_plot = EDA.EDA(st.session_state[optionDataset])
                    if var_graph_outlier != 'None':
                        def plot_px_boxplot(obj_plot, col_y):
                            st.subheader('Boxplot')
                            st.plotly_chart(obj_plot.box_plot(col_y))
                        
                        plot_px_boxplot(eda_plot, var_graph_outlier)
                        st.write("-------------------------")

                show_limits = st.checkbox("Show limits")
                if show_limits:

                    if var_graph_outlier:
                        
                        st.dataframe(st.session_state[optionDataset][var_graph_outlier].describe())
                        
                        for col in var_graph_outlier:
                            inf_limit = st.session_state[optionDataset][col].quantile(0.25) - 1.5 * (st.session_state[optionDataset][col].quantile(0.75) - st.session_state[optionDataset][col].quantile(0.25))
                            over_limit = st.session_state[optionDataset][col].quantile(0.75) + 1.5 * (st.session_state[optionDataset][col].quantile(0.75) - st.session_state[optionDataset][col].quantile(0.25))
                            st.write(
                            f"""

                            {col}:
                            - Inferior limit = {st.session_state[optionDataset][col].quantile(0.25)} - 1.5 * ({st.session_state[optionDataset][col].quantile(0.75)} - {st.session_state[optionDataset][col].quantile(0.25)}) |> **Values less than {inf_limit} is outlier**
                            - Over limit = {st.session_state[optionDataset][col].quantile(0.75)} + 1,5 * ({st.session_state[optionDataset][col].quantile(0.75)} - {st.session_state[optionDataset][col].quantile(0.25)}) |> **Values greater than {over_limit} is outlier**
                            - **{len(st.session_state[optionDataset][(st.session_state[optionDataset][col] < inf_limit) | (st.session_state[optionDataset][col] > over_limit)])}** instance with values outliers in {col} column
                            ---------------------------------------------------------------
                            """
                            )

                    else:
                        st.warning("Select the variable!")

                st.subheader("Modify outliers")

                var_select_opt_outlier  = st.selectbox("Select the variable", st.session_state[optionDataset].select_dtypes(include=[np.number]).columns, key ='var_select_opt_outlier')
                delete_or_imputation_outlier = st.selectbox("Exclude or impute outliers?",['None','Imputation','Exclude'])
                if delete_or_imputation_outlier == 'Exclude':
                    submit_outlier = st.button("Apply", key='submit_outlier')
                    code_outlier_detection = \
f'''inf_limit = self.base['{var_select_opt_outlier}'].quantile(0.25) - 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
over_limit = self.base['{var_select_opt_outlier}'].quantile(0.75) + 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
self.base = self.base[(self.base['{var_select_opt_outlier}'] > inf_limit) & (self.base['{var_select_opt_outlier}'] < over_limit)]'''
                    st.code(code_outlier_detection, language='python')
                    if submit_outlier:
                        inf_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25) - 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                        over_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) + 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                        st.session_state[optionDataset] = st.session_state[optionDataset][(st.session_state[optionDataset][var_select_opt_outlier] > inf_limit) & (st.session_state[optionDataset][var_select_opt_outlier] < over_limit)]
                        st.experimental_rerun()
                elif delete_or_imputation_outlier == 'Imputation':
                    imputation_outlier = st.selectbox("Select the imputation",['None','Mean','Median','Manual'])
                    if imputation_outlier == 'Mean':
                        submit_outlier = st.button("Apply", key='submit_outlier')
                        code_outlier_detection = \
f'''inf_limit = self.base['{var_select_opt_outlier}'].quantile(0.25) - 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
over_limit = self.base['{var_select_opt_outlier}'].quantile(0.75) + 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
self.base['{var_select_opt_outlier}'] = self.base['{var_select_opt_outlier}'].apply(lambda num: self.base['{var_select_opt_outlier}'].mean() if num < inf_limit or num > over_limit else num)'''
                        st.code(code_outlier_detection, language='python')
                        if submit_outlier:
                            inf_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25) - 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            over_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) + 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            st.session_state[optionDataset][var_select_opt_outlier] = st.session_state[optionDataset][var_select_opt_outlier].apply(lambda num: st.session_state[optionDataset][var_select_opt_outlier].mean() if num < inf_limit or num > over_limit else num)
                            st.experimental_rerun()
                    elif imputation_outlier == 'Median':
                        submit_outlier = st.button("Apply", key='submit_outlier')
                        code_outlier_detection = \
f'''inf_limit = self.base['{var_select_opt_outlier}'].quantile(0.25) - 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
over_limit = self.base['{var_select_opt_outlier}'].quantile(0.75) + 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
self.base['{var_select_opt_outlier}'] = self.base['{var_select_opt_outlier}'].apply(lambda num: self.base['{var_select_opt_outlier}'].median() if num < inf_limit or num > over_limit else num)'''
                        st.code(code_outlier_detection, language='python')
                        if submit_outlier:
                            inf_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25) - 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            over_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) + 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            st.session_state[optionDataset][var_select_opt_outlier] = st.session_state[optionDataset][var_select_opt_outlier].apply(lambda num: st.session_state[optionDataset][var_select_opt_outlier].median() if num < inf_limit or num > over_limit else num)
                            st.experimental_rerun()
                    elif imputation_outlier == 'Manual':
                        input_number_manual_imputation = st.number_input("Input the value to imputation")
                        submit_outlier = st.button("Apply", key='submit_outlier')
                        code_outlier_detection = \
f'''inf_limit = self.base['{var_select_opt_outlier}'].quantile(0.25) - 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
over_limit = self.base['{var_select_opt_outlier}'].quantile(0.75) + 1.5 * (self.base['{var_select_opt_outlier}'].quantile(0.75) - self.base['{var_select_opt_outlier}'].quantile(0.25))
self.base['{var_select_opt_outlier}'] = self.base['{var_select_opt_outlier}'].apply(lambda num: {input_number_manual_imputation} if num < inf_limit or num > over_limit else num)'''
                        st.code(code_outlier_detection, language='python')
                        if submit_outlier:
                            inf_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25) - 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            over_limit = st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) + 1.5 * (st.session_state[optionDataset][var_select_opt_outlier].quantile(0.75) - st.session_state[optionDataset][var_select_opt_outlier].quantile(0.25))
                            st.session_state[optionDataset][var_select_opt_outlier] = st.session_state[optionDataset][var_select_opt_outlier].apply(lambda num: input_number_manual_imputation if num < inf_limit or num > over_limit else num)
                            st.experimental_rerun()

            with st.expander("Principal Component Analysis (PCA)", expanded=False):

                PCAType = option_menu('', ['Create PCA','Apply transform'], 
                    default_index=0, orientation="horizontal",
                    styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                            "nav-link": {"font-size": "13px","--hover-color": "#eee"}}, key='sub_menu_PCA'
                )
                
                if PCAType == 'Create PCA':
                    
                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        var_pca = st.multiselect("Select the variable to apply the pca", st.session_state[optionDataset].columns)
                    checkbox_explained_variance = st.checkbox("Show explained variance")
                    if checkbox_explained_variance:
                        if var_pca:
                            import plotly.express as px
                            from sklearn.decomposition import PCA

                            df_pca = pd.DataFrame(preprocessing.StandardScaler().fit_transform(st.session_state[optionDataset][var_pca]), columns=var_pca)
                            pca = PCA()
                            pca.fit(df_pca)
                            total_components = len(pca.explained_variance_ratio_)
                            name_cols_pca_explained = ["Component_PCA_"+str(col) for col in range(1, total_components+1)]
                            st.write(list(zip(name_cols_pca_explained,pca.explained_variance_ratio_)))
                            exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

                            st.plotly_chart(px.area(
                                x=range(1, exp_var_cumul.shape[0] + 1),
                                y=exp_var_cumul,
                                labels={"x": "# Components", "y": "Explained Variance"}
                            ))

                    if var_pca:
                        st.subheader("Apply PCA in variables")

                        col1, col2 = st.columns([3,2])
                        with col1:
                            save_object_pca = st.checkbox("Save object", key='save_object_pca')

                        col1, col2 = st.columns([3,2])
                        with col1:
                            if save_object_pca:
                                name_object_pca = st.text_input("Input object name to save", key='name_object_pca')

                        col1, col2 = st.columns([3,2])
                        with col1:
                            select_opt_to_save_pca = st.selectbox("Select option to save",['None','Join the components in the dataset and delete the variables that formed the pca',
                                                                                        'Join the components in the dataset and no delete the variables that formed the pca',
                                                                                        'Save the result in a new dataset'])

                        col1, col2 = st.columns([1.5,3.5])
                        with col1:
                            standardscaler_pca = st.selectbox("Apply StandardScaler before",['Yes','No'])

                        col1, col2 = st.columns([1.5,3.5])
                        with col1:
                            name_pca = st.text_input("Input components name")

                        col1, col2 = st.columns([1.5,3.5])
                        with col1:
                            select_n_components = st.number_input("Select number of components", value=0, step=1, min_value=0)

                        if select_opt_to_save_pca == 'Join the components in the dataset and delete the variables that formed the pca' and select_n_components != 0:
                            submit_apply_pca = st.button("Apply PCA", key='submit_apply_pca')
                            if standardscaler_pca == 'Yes':
                                if save_object_pca and name_object_pca != '':
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
scaler_pca = preprocessing.StandardScaler()
df_pca = pd.DataFrame(scaler_pca.fit(self.base[{var_pca}]).transform(self.base[{var_pca}]), columns={var_pca})
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base.drop({var_pca},axis=1,inplace=True)
self.base = pd.concat([self.base,df_X_pca],axis=1)
pickle.dump(scaler_pca, open(f'saved_models/{name_object_pca}_pcaScaler.pkl','wb'))
pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))'''
                                    st.code(code_pca, language='python')
                                else:
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
scaler_pca = preprocessing.StandardScaler()
df_pca = pd.DataFrame(scaler_pca.fit(self.base[{var_pca}]).transform(self.base[{var_pca}]), columns={var_pca})
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base.drop({var_pca},axis=1,inplace=True)
self.base = pd.concat([self.base,df_X_pca],axis=1)'''
                                    st.code(code_pca, language='python')
                            else:
                                if save_object_pca and name_object_pca != '':
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> import pickle
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
df_pca = self.base[{var_pca}]
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base.drop({var_pca},axis=1,inplace=True)
self.base = pd.concat([self.base,df_X_pca],axis=1)
pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))'''
                                    st.code(code_pca, language='python')
                                else:
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
df_pca = self.base[{var_pca}]
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base.drop({var_pca},axis=1,inplace=True)
self.base = pd.concat([self.base,df_X_pca],axis=1)'''
                                    st.code(code_pca, language='python')

                            if submit_apply_pca:
                                from sklearn.decomposition import PCA
                                name_cols_pca = [f"{name_pca}_PCA_"+str(col) for col in range(1, select_n_components+1)]
                                if standardscaler_pca == 'Yes':
                                    if save_object_pca and name_object_pca != '':
                                        scaler_pca = preprocessing.StandardScaler()
                                        df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)
                                        
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset].drop(var_pca,axis=1,inplace=True)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write(print("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_))))
                                    
                                        pickle.dump(scaler_pca, open(f'saved_models/{name_object_pca}_pcaScaler.pkl','wb'))
                                        pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                    else:
                                        scaler_pca = preprocessing.StandardScaler()
                                        df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)
                                        
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset].drop(var_pca,axis=1,inplace=True)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write(print("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_))))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                
                                else:
                                    if save_object_pca and name_object_pca != '':
                                        df_pca = st.session_state[optionDataset][var_pca]
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset].drop(var_pca,axis=1,inplace=True)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write(print("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_))))
                                
                                        pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                    else:
                                        df_pca = st.session_state[optionDataset][var_pca]
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset].drop(var_pca,axis=1,inplace=True)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write(print("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_))))
                                        time.sleep(1)
                                        st.experimental_rerun()

                        elif select_opt_to_save_pca == 'Join the components in the dataset and no delete the variables that formed the pca' and select_n_components != 0:
                            submit_apply_pca = st.button("Apply PCA", key='submit_apply_pca')
                            if standardscaler_pca == 'Yes':
                                if save_object_pca and name_object_pca != '':
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
scaler_pca = preprocessing.StandardScaler()
df_pca = pd.DataFrame(scaler_pca.fit(self.base[{var_pca}]).transform(self.base[{var_pca}]), columns={var_pca})
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base = pd.concat([self.base,df_X_pca],axis=1)
pickle.dump(scaler_pca, open(f'saved_models/{name_object_pca}_pcaScaler.pkl','wb'))
pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))'''
                                    st.code(code_pca, language='python')
                                else:
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
scaler_pca = preprocessing.StandardScaler()
df_pca = pd.DataFrame(scaler_pca.fit(self.base[{var_pca}]).transform(self.base[{var_pca}]), columns={var_pca})
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base = pd.concat([self.base,df_X_pca],axis=1)'''
                                    st.code(code_pca, language='python')
                            else:
                                if save_object_pca and name_object_pca != '':
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
# import the lib -> import pickle
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
df_pca = self.base[{var_pca}]
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base = pd.concat([self.base,df_X_pca],axis=1)
pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))'''
                                    st.code(code_pca, language='python')
                                else:
                                    code_pca = \
f'''# import the lib -> from sklearn.decomposition import PCA
# import the lib -> from sklearn import preprocessing
name_cols_pca = ["{name_pca}_PCA_"+str(col) for col in range(1, {select_n_components}+1)]
df_pca = self.base[{var_pca}]
pca = PCA(n_components={select_n_components}).fit(df_pca)
X_components = pca.transform(df_pca)
df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
self.base = pd.concat([self.base,df_X_pca],axis=1)'''
                                    st.code(code_pca, language='python')

                            if submit_apply_pca:
                                from sklearn.decomposition import PCA
                                name_cols_pca = [f"{name_pca}_PCA_"+str(col) for col in range(1, select_n_components+1)]
                                if standardscaler_pca == 'Yes':
                                    if save_object_pca and name_object_pca != '':
                                        scaler_pca = preprocessing.StandardScaler()
                                        df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)   
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))

                                        pickle.dump(scaler_pca, open(f'saved_models/{name_object_pca}_pcaScaler.pkl','wb'))
                                        pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                    else:
                                        scaler_pca = preprocessing.StandardScaler()
                                        df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)   
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                
                                else:
                                    if save_object_pca and name_object_pca != '':
                                        df_pca = st.session_state[optionDataset][var_pca]
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))

                                        pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                        time.sleep(1)
                                        st.experimental_rerun()
                                    else:
                                        df_pca = st.session_state[optionDataset][var_pca]
                                        pca = PCA(n_components=select_n_components).fit(df_pca)
                                        X_components = pca.transform(df_pca)
                                        df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                        st.session_state[optionDataset] = pd.concat([st.session_state[optionDataset],df_X_pca],axis=1)
                                        #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))
                                        time.sleep(1)
                                        st.experimental_rerun()

                        elif select_opt_to_save_pca == 'Save the result in a new dataset' and select_n_components != 0:
                            select_the_object_to_save_pca = st.selectbox("Select the object", st.session_state['Objects'])
                            if not select_the_object_to_save_pca in st.session_state['dataset']:
                                submit_apply_pca = st.button("Apply PCA", key='submit_apply_pca')
                                st.warning("Don't have code!")
                                if submit_apply_pca:
                                    from sklearn.decomposition import PCA
                                    name_cols_pca = [f"{name_pca}_PCA_"+str(col) for col in range(1, select_n_components+1)]
                                    if standardscaler_pca == 'Yes':
                                        if save_object_pca and name_object_pca != '':
                                            scaler_pca = preprocessing.StandardScaler()
                                            df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)
                                            pca = PCA(n_components=select_n_components).fit(df_pca)
                                            X_components = pca.transform(df_pca)
                                            df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                            st.session_state[select_the_object_to_save_pca] = df_X_pca
                                            st.session_state['dataset'].append(select_the_object_to_save_pca)
                                            #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))
                                        
                                            pickle.dump(scaler_pca, open(f'saved_models/{name_object_pca}_pcaScaler.pkl','wb'))
                                            pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                            time.sleep(1)
                                            st.experimental_rerun()
                                        else:
                                            scaler_pca = preprocessing.StandardScaler()
                                            df_pca = pd.DataFrame(scaler_pca.fit(st.session_state[optionDataset][var_pca]).transform(st.session_state[optionDataset][var_pca]), columns=var_pca)
                                            pca = PCA(n_components=select_n_components).fit(df_pca)
                                            X_components = pca.transform(df_pca)
                                            df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                            st.session_state[select_the_object_to_save_pca] = df_X_pca
                                            st.session_state['dataset'].append(select_the_object_to_save_pca)
                                            #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))
                                            time.sleep(1)
                                            st.experimental_rerun()
                                    
                                    else:
                                        if save_object_pca and name_object_pca != '':
                                            df_pca = st.session_state[optionDataset][var_pca]
                                            pca = PCA(n_components=select_n_components).fit(df_pca)
                                            X_components = pca.transform(df_pca)
                                            df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                            st.session_state[select_the_object_to_save_pca] = df_X_pca
                                            st.session_state['dataset'].append(select_the_object_to_save_pca)
                                            #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))

                                            pickle.dump(pca, open(f'saved_models/{name_object_pca}.pkl','wb'))
                                            time.sleep(1)
                                            st.experimental_rerun()
                                        else:
                                            df_pca = st.session_state[optionDataset][var_pca]
                                            pca = PCA(n_components=select_n_components).fit(df_pca)
                                            X_components = pca.transform(df_pca)
                                            df_X_pca = pd.DataFrame(X_components, columns=name_cols_pca)
                                            st.session_state[select_the_object_to_save_pca] = df_X_pca
                                            st.session_state['dataset'].append(select_the_object_to_save_pca)
                                            #st.write("Variability:",list(zip(name_cols_pca,pca.explained_variance_ratio_)))
                                            time.sleep(1)
                                            st.experimental_rerun()
                            else:
                                st.error("This object is already in use, create a new object")

                elif PCAType == 'Apply transform':
                    
                    pca = None
                    name_tpca = ''
                    options_files_scaler_pca = None
                    options_files_pca = None
                    select_data_to_apply_tpca = None
                    columns_to_tpca = None
                    insert_or_select_columns_tpca = None
                    code_tpca = None

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        with_scaler = st.selectbox("Has StandardScaler?",['None','Yes','No'])

                    if with_scaler == 'Yes':
                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        col1, col2, col3 = st.columns([1.5,1.5,2])
                        with col1:
                            options_files_scaler_pca = st.selectbox('Select scaler object', list_files)
                            if options_files_scaler_pca != 'None':
                                scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
                        with col2:
                            options_files_pca = st.selectbox('Select pca object', list_files, key='options_files_pca')
                            if options_files_pca != 'None':
                                pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))

                    elif with_scaler == 'No':
                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        col1, col2, col3 = st.columns([1.5,1.5,2])
                        with col1:
                            options_files_pca = st.selectbox('Select pca object', list_files, key='options_files_pca')
                            if options_files_pca != 'None':
                                pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if options_files_scaler_pca or options_files_pca:
                            name_tpca = st.text_input("Input PCA name")

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if name_tpca != '':
                            select_data_to_apply_tpca = st.selectbox("Select dataset", st.session_state['dataset'])

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if select_data_to_apply_tpca:
                            insert_or_select_columns_tpca = st.selectbox("Insert or select columns?",['None','Insert','Select'], key='insert_or_select_columns_tpca')

                    col1, col2 = st.columns([1.5,3.5])
                    with col1:
                        if insert_or_select_columns_tpca == 'Insert':
                            insert_columns_to_tpca = st.text_input("Insert the columns to apply transform")
                            columns_to_tpca = insert_columns_to_tpca.split(',')
                            keep_or_delete_columns_tpca = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_tpca')
                            if insert_columns_to_tpca:
                                if keep_or_delete_columns_tpca == 'Yes':
                                    if with_scaler and options_files_scaler_pca and options_files_pca:
                                        code_tpca = \
f'''# import the lib -> import pickle
scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_std = scaler_pca.transform(self.base[{columns_to_tpca}])
df_pca = pca.transform(df_std)
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base.drop({columns_to_tpca},axis=1), df_pca], axis=1)'''
                                    else:
                                        code_tpca = \
f'''# import the lib -> import pickle
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_pca = pca.transform(self.base[{columns_to_tpca}])
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base.drop({columns_to_tpca},axis=1), df_mms], axis=1)'''


                                else:
                                    if with_scaler and options_files_scaler_pca and options_files_pca:
                                        code_tpca = \
f'''# import the lib -> import pickle
scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_std = scaler_pca.transform(self.base[{columns_to_tpca}])
df_pca = pca.transform(df_std)
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base, df_pca], axis=1)'''
                                    else:
                                        code_tpca = \
f'''# import the lib -> import pickle
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_pca = pca.transform(self.base[{columns_to_tpca}])
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base, df_pca], axis=1)'''


                        elif insert_or_select_columns_tpca == 'Select':
                            columns_to_tpca = st.multiselect("Select the columns to apply transform", st.session_state[select_data_to_apply_tpca].columns)
                            keep_or_delete_columns_tpca = st.selectbox("Delete transformed column",['Yes','No'], key='keep_or_delete_columns_tpca')
                            if columns_to_tpca:
                                if keep_or_delete_columns_tpca == 'Yes':
                                    if with_scaler and options_files_scaler_pca and options_files_pca:
                                        code_tpca = \
f'''# import the lib -> import pickle
scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_std = scaler_pca.transform(self.base[{columns_to_tpca}])
df_pca = pca.transform(df_std)
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base.drop({columns_to_tpca},axis=1), df_pca], axis=1)'''
                                    else:
                                        code_tpca = \
f'''# import the lib -> import pickle
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_pca = pca.transform(self.base[{columns_to_tpca}])
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base.drop({columns_to_tpca},axis=1), df_mms], axis=1)'''

                                else:
                                    if with_scaler and options_files_scaler_pca and options_files_pca:
                                        code_tpca = \
f'''# import the lib -> import pickle
scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_std = scaler_pca.transform(self.base[{columns_to_tpca}])
df_pca = pca.transform(df_std)
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base, df_pca], axis=1)'''
                                    else:
                                        code_tpca = \
f'''# import the lib -> import pickle
pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
df_pca = pca.transform(self.base[{columns_to_tpca}])
name_col_pca = [col + '_pca' for col in self.base[{columns_to_tpca}].columns]
df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
self.base = pd.concat([self.base, df_pca], axis=1)'''

                    if code_tpca:
                        st.code(code_tpca, language='python')

                    if pca and select_data_to_apply_tpca and columns_to_tpca:
                        button_apply_tpca = st.button("Apply", key='button_apply_tpca')
                        if button_apply_tpca:
                            if keep_or_delete_columns_tpca == 'Yes':
                                if with_scaler and options_files_scaler_pca and options_files_pca:
                                    #name_pca = 'teste22'
                                    #scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
                                    #pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
                                    df_std = scaler_pca.transform(st.session_state[select_data_to_apply_tpca][columns_to_tpca])
                                    df_pca = pca.transform(df_std)
                                    #name_col_pca = [col + '_pca' for col in st.session_state[select_data_to_apply_tpca][columns_to_tpca].columns]
                                    name_col_pca = [f"{name_tpca}_PCA_"+str(col) for col in range(1, pca.n_components_ + 1)]
                                    df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
                                    st.session_state[select_data_to_apply_tpca] = pd.concat([st.session_state[select_data_to_apply_tpca].drop(columns_to_tpca,axis=1), df_pca], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    #name_pca = 'teste22'
                                    #pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
                                    df_pca = pca.transform(st.session_state[select_data_to_apply_tpca][columns_to_tpca])
                                    #name_col_pca = [col + '_pca' for col in st.session_state[select_data_to_apply_tpca][columns_to_tpca].columns]
                                    name_col_pca = [f"{name_tpca}_PCA_"+str(col) for col in range(1, pca.n_components_ + 1)]
                                    df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
                                    st.session_state[select_data_to_apply_tpca] = pd.concat([st.session_state[select_data_to_apply_tpca].drop(columns_to_tpca,axis=1), df_pca], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                            else:
                                if with_scaler and options_files_scaler_pca and options_files_pca:
                                    #name_pca = 'teste22'
                                    #scaler_pca = pickle.load(open(f'saved_models/{options_files_scaler_pca}','rb'))
                                    #pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
                                    df_std = scaler_pca.transform(st.session_state[select_data_to_apply_tpca][columns_to_tpca])
                                    df_pca = pca.transform(df_std)
                                    #name_col_pca = [col + '_pca' for col in st.session_state[select_data_to_apply_tpca][columns_to_tpca].columns]
                                    name_col_pca = [f"{name_tpca}_PCA_"+str(col) for col in range(1, pca.n_components_ + 1)]
                                    df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
                                    st.session_state[select_data_to_apply_tpca] = pd.concat([st.session_state[select_data_to_apply_tpca], df_pca], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()
                                else:
                                    #name_pca = 'teste22'
                                    #pca = pickle.load(open(f'saved_models/{options_files_pca}','rb'))
                                    df_pca = pca.transform(st.session_state[select_data_to_apply_tpca][columns_to_tpca])
                                    #name_col_pca = [col + '_pca' for col in st.session_state[select_data_to_apply_tpca][columns_to_tpca].columns]
                                    name_col_pca = [f"{name_tpca}_PCA_"+str(col) for col in range(1, pca.n_components_ + 1)]
                                    df_pca = pd.DataFrame(df_pca, columns=name_col_pca)
                                    st.session_state[select_data_to_apply_tpca] = pd.concat([st.session_state[select_data_to_apply_tpca], df_pca], axis=1)
                                    time.sleep(1)
                                    st.experimental_rerun()

            with st.expander("Imbalanced data", expanded=False):

                st.image(Image.open('images/over_under_sampling.png'), width=600)

                st.warning("The target variable must be of type numeric")

                code_imbalanced = None
                select_method = None

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    data_separated = st.selectbox("The data is separated?",['No','Yes'])

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    if data_separated == 'No':
                        select_target_var = st.selectbox("Select the variable target", options_columns)
                    else:
                        x_imbalanced_data = st.selectbox("Select the dataset(X)", st.session_state['dataset'])
                with col2:
                    if data_separated == 'Yes':
                        y_imbalanced_data = st.selectbox("Select the target(y)", st.session_state['dataset'])

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:       
                    select_technique = st.selectbox("Select balance type", ['None','Oversampling','Undersampling'])
                with col2:
                    if select_technique == 'Oversampling':
                        select_method = st.selectbox("Select the method", ['None','RandomOverSampler','SMOTE','ADASYN'])
                    elif select_technique == 'Undersampling':
                        select_method = st.selectbox("Select the method", ['None','NearMiss','RandomUnderSampler','TomekLinks'])
                with col3:
                    input_strategy = None
                    if select_technique != 'None':
                        input_strategy = st.selectbox("Input sampling_strategy?",['No','Yes'])
                with col4:
                    if input_strategy == 'Yes':
                        opt_strategy = st.number_input("Input number of sampling_strategy", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
                    else:
                        opt_strategy = 'auto'

                if select_method == 'RandomOverSampler':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import RandomOverSampler
method_imbalanced = RandomOverSampler(sampling_strategy={opt_strategy}, random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import RandomOverSampler
method_imbalanced = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                elif select_method == 'SMOTE':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import SMOTE
method_imbalanced = SMOTE(sampling_strategy={opt_strategy}, random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import SMOTE
method_imbalanced = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                elif select_method == 'ADASYN':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import ADASYN
method_imbalanced = ADASYN(sampling_strategy={opt_strategy}, random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.over_sampling import ADASYN
method_imbalanced = ADASYN(sampling_strategy='auto', random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                elif select_method == 'NearMiss':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import NearMiss
method_imbalanced = NearMiss(sampling_strategy={opt_strategy})
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import NearMiss
method_imbalanced = NearMiss(sampling_strategy='auto')
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                elif select_method == 'RandomUnderSampler':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import RandomUnderSampler
method_imbalanced = RandomUnderSampler(sampling_strategy={opt_strategy}, random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import RandomUnderSampler
method_imbalanced = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                elif select_method == 'TomekLinks':
                    if input_strategy == 'Yes':
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import TomekLinks
method_imbalanced = TomekLinks(sampling_strategy={opt_strategy})
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''
                    else:
                        if data_separated == 'No':
                            code_imbalanced = \
f'''# import the lib -> from imblearn.under_sampling import TomekLinks
method_imbalanced = TomekLinks(sampling_strategy='auto')
X_res, y_res = method_imbalanced.fit_resample(self.base.drop(['{select_target_var}'],axis=1), self.base['{select_target_var}'].astype('int'))
self.base = pd.concat([X_res, y_res],axis=1)'''

                if code_imbalanced:
                    st.code(code_imbalanced, language='python')

                if select_technique == 'Oversampling' and select_method != 'None':
                    st.info("While the RandomOverSampler is over-sampling by duplicating some of the original samples of the minority class, SMOTE and ADASYN generate new samples in by interpolation. However, the samples used to interpolate/generate new synthetic samples differ. In fact, ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier while the basic implementation of SMOTE will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule. Therefore, the decision function found during training will be different among the algorithms.")
                    st.image(Image.open('images/graph_smote_adasyn.png'), width=700)

                if select_technique != 'None' and select_method != 'None':

                    button_view_imbalanced = st.button("Apply", key='button_view_imbalanced')
                    if button_view_imbalanced:
                        import plotly.express as px
                        with st.spinner('Wait for it...'):
                            
                            if select_method == 'RandomOverSampler':
                                from imblearn.over_sampling import RandomOverSampler
                                method_imbalanced = RandomOverSampler(sampling_strategy=opt_strategy, random_state=42)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'RandomOverSampler'
                                    if not optionDataset+'_resampling_RandomOverSampler' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'RandomOverSampler'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_RandomOverSampler' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_RandomOverSampler' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_RandomOverSampler'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_RandomOverSampler'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_RandomOverSampler')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_RandomOverSampler')
                                    del X_res
                                    del y_res

                            elif select_method == 'SMOTE':
                                from imblearn.over_sampling import SMOTE
                                method_imbalanced = SMOTE(sampling_strategy=opt_strategy, random_state=42)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'SMOTE'
                                    if not optionDataset+'_resampling_SMOTE' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'SMOTE'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_SMOTE' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_SMOTE' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_SMOTE'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_SMOTE'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_SMOTE')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_SMOTE')
                                    del X_res
                                    del y_res

                            elif select_method == 'ADASYN':
                                from imblearn.over_sampling import ADASYN
                                method_imbalanced = ADASYN(sampling_strategy=opt_strategy, random_state=42)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'ADASYN'
                                    if not optionDataset+'_resampling_ADASYN' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'ADASYN'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_ADASYN' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_ADASYN' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_ADASYN'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_ADASYN'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_ADASYN')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_ADASYN')
                                    del X_res
                                    del y_res

                            elif select_method == 'NearMiss':
                                from imblearn.under_sampling import NearMiss
                                method_imbalanced = NearMiss(sampling_strategy=opt_strategy)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'NearMiss'
                                    if not optionDataset+'_resampling_NearMiss' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'NearMiss'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_NearMiss' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_NearMiss' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_NearMiss'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_NearMiss'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_NearMiss')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_NearMiss')
                                    del X_res
                                    del y_res

                            elif select_method == 'RandomUnderSampler':
                                from imblearn.under_sampling import RandomUnderSampler
                                method_imbalanced = RandomUnderSampler(sampling_strategy=opt_strategy, random_state=42)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'RandomUnderSampler'
                                    if not optionDataset+'_resampling_RandomUnderSampler' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'RandomUnderSampler'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_RandomUnderSampler' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_RandomUnderSampler' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_RandomUnderSampler'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_RandomUnderSampler'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_RandomUnderSampler')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_RandomUnderSampler')
                                    del X_res
                                    del y_res

                            elif select_method == 'TomekLinks':
                                from imblearn.under_sampling import TomekLinks
                                method_imbalanced = TomekLinks(sampling_strategy=opt_strategy)
                                if data_separated == 'No':
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[optionDataset].drop([select_target_var],axis=1), st.session_state[optionDataset][select_target_var].astype('int'))
                                    complet_name = 'TomekLinks'
                                    if not optionDataset+'_resampling_TomekLinks' in st.session_state['dataset']:
                                        st.session_state[optionDataset+'_resampling_'+complet_name] = pd.concat([X_res, y_res],axis=1)
                                        st.session_state['dataset'].append(optionDataset+'_resampling_'+complet_name)
                                    del X_res
                                    del y_res
                                else:
                                    complet_name = 'TomekLinks'
                                    X_res, y_res = method_imbalanced.fit_resample(st.session_state[x_imbalanced_data], st.session_state[y_imbalanced_data].astype('int'))
                                    if not x_imbalanced_data+'_resampling_TomekLinks' in st.session_state['dataset']:
                                        if not y_imbalanced_data+'_resampling_TomekLinks' in st.session_state['dataset']:
                                            st.session_state[x_imbalanced_data+'_resampling_TomekLinks'] = X_res
                                            st.session_state[y_imbalanced_data+'_resampling_TomekLinks'] = y_res
                                            st.session_state['dataset'].append(x_imbalanced_data+'_resampling_TomekLinks')
                                            st.session_state['dataset'].append(y_imbalanced_data+'_resampling_TomekLinks')
                                    del X_res
                                    del y_res

                            st.title('Resampling results')

                            if data_separated == 'No':

                                st.subheader("Histogram with target distribution")

                                col1_imbalanced_results_1, col2_imbalanced_results_1 = st.columns([0.5,0.5])
                                with col1_imbalanced_results_1:
                                    fig = px.histogram(st.session_state[optionDataset], x=select_target_var, color=select_target_var, title='Before resamplings')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2_imbalanced_results_1:
                                    fig = px.histogram(st.session_state[optionDataset+'_resampling_'+complet_name], x=select_target_var, color=select_target_var, title='After resamplings')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)

                                st.subheader("Data description")

                                st.write("Before resampling")
                                st.dataframe(st.session_state[optionDataset].describe())
                                st.write("After resampling")
                                st.dataframe(st.session_state[optionDataset+'_resampling_'+complet_name].describe())

                            else:
                                #st.warning("Coming soon!")
                                st.subheader("Histogram with target distribution")

                                col1_imbalanced_results_1, col2_imbalanced_results_1 = st.columns([0.5,0.5])
                                with col1_imbalanced_results_1:
                                    fig = px.histogram(st.session_state[y_imbalanced_data], x=st.session_state[y_imbalanced_data].columns[0], color=st.session_state[y_imbalanced_data].columns[0], title='Before resamplings')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2_imbalanced_results_1:
                                    fig = px.histogram(st.session_state[y_imbalanced_data+'_resampling_'+complet_name], x=st.session_state[y_imbalanced_data+'_resampling_'+complet_name].columns[0], color=st.session_state[y_imbalanced_data+'_resampling_'+complet_name].columns[0], title='After resamplings')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)


            st.write("-------------------------")

            with st.expander("Code in script", expanded=False):

                button_update_code_script = st.button("Refresh page", key='button_refreshpage_code_script_1')
                if button_update_code_script:
                    st.experimental_rerun()
                
                with open("libs/script_to_run_out.py", "r") as f:
                    file_contents = f.read()

                content = st_ace(file_contents,language='python', theme='pastel_on_dark')

                col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
                with col_code_script_1:
                    opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                    if opt_code_script == 'Save script':
                        button_code_save_script = st.button("Save", key='button_code_save_script')
                        if button_code_save_script:
                            file = open("libs/script_to_run_out.py", "w") 
                            file.write(content)
                            file.close()
                            time.sleep(1)
                            st.experimental_rerun()
                    elif opt_code_script == 'Download script':
                        st.download_button(
                            label="Download script",
                            data=file_contents,
                            file_name='script_to_run_out.py'
                        )
                    elif opt_code_script == 'Reset script':
                        button_code_reset_script = st.button("Reset", key='button_code_reset_script_1')
                        if button_code_reset_script:
                            with open("libs/script_to_run_out_backup.py", "r") as f:
                                file_contents_backup = f.read()

                            file = open("libs/script_to_run_out.py", "w") 
                            file.write(file_contents_backup)
                            file.close()
                            time.sleep(1)
                            st.experimental_rerun()

            with st.expander("View dataset information",expanded=False):

                #@st.cache
                def func_tab_final():
                    if list(st.session_state[optionDataset].select_dtypes(include=[np.number]).columns):
                        tab_numbers = st.session_state[optionDataset].describe(include=[np.number]).T
                    else:
                        tab_numbers = pd.DataFrame()
                    if list(st.session_state[optionDataset].select_dtypes(include=['object','category']).columns):
                        tab_objects = st.session_state[optionDataset].describe(include=['object','category']).T
                    else:
                        tab_objects = pd.DataFrame()

                    tab_joins = pd.concat([tab_objects, tab_numbers],axis=1)

                    tab_nans = pd.DataFrame(st.session_state[optionDataset].isna().sum(), columns=['NaN'])
                    tab_nans['NaN%'] = round((st.session_state[optionDataset].isna().sum() / st.session_state[optionDataset].shape[0])*100,2)

                    tab = tab_joins.join(tab_nans)

                    dfx_dtypes = pd.DataFrame(st.session_state[optionDataset].dtypes, columns=['type'])
                    dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                    del dfx_dtypes['type']

                    tab_final = tab.join(dfx_dtypes)
                    if tab_objects.shape[0] > 0 and tab_numbers.shape[0] > 0:
                        tab_final.columns = ['count', 'unique', 'top', 'freq', 'count_number', 'mean', 'std', 
                            'min', '25%','50%', '75%', 'max', 'NaN', 'NaN%', 'type_str']

                        tab_final['count'] = np.where(tab_final['count'].isnull(), tab_final['count_number'], tab_final['count'])
                        tab_final['count'] = tab_final['count'].astype(int)
                        del tab_final['count_number']
                    return tab_final

                col1, col2, col3 = st.columns([0.8,0.8,2.5])
                with col1:
                    init_rows = st.number_input("Min. of rows", step=1, value=0, min_value=0)
                with col2:
                    final_rows = st.number_input("Max. of rows", step=1, value=st.session_state[optionDataset].shape[0] if st.session_state[optionDataset].shape[0] < 5 else 5, min_value=0, max_value=int(st.session_state[optionDataset].shape[0]))
                    
                col11, col22, col33 = st.columns([0.8,0.8,2.5])
                with col11:
                    init_cols = st.number_input("Min. of cols", step=1, value=0, min_value=0)
                with col22:
                    final_cols = st.number_input("Max. of cols", step=1, value=int(st.session_state[optionDataset].shape[1]), min_value=0, max_value=int(st.session_state[optionDataset].shape[1]))

                selectionColumn = st.checkbox('Columns selections')

                st.header("View dataset")
                if selectionColumn:
                    optionsViewColumns = st.multiselect("Multiselect columns",list(st.session_state[optionDataset].columns))
                    #st.write(optionsViewColumns)
                    st.write(
                        f"""
                        **Columns selected**: {", ".join(optionsViewColumns)}
                        """
                    )
                    st.dataframe(st.session_state[optionDataset][optionsViewColumns].iloc[int(init_rows):int(final_rows)])
                else:
                    st.dataframe(st.session_state[optionDataset].iloc[int(init_rows):int(final_rows), int(init_cols):int(final_cols)])

                st.write("------------------------------------------------------")
                st.header("Descriptive")
                show_me_descriptive = st.checkbox("Show descriptive")
                if show_me_descriptive:
                    st.dataframe(func_tab_final())

        else:

            st.write("There is no Dataset loaded")
            
    def appExecuteScript():

        st.title('Execute script in object')

        with st.expander("Click here for more info on this app section", expanded=False):

            st.write(
            f"""

            Summary
            ---------------
            Welcome to the application's scripting session. So that you can have more options for processing your dataset,
            you can run functions directly from a python script.

            **Select the dataset and proceed with the execution**
            """
            )

        with st.expander("Code in script", expanded=False):

            button_update_code_script = st.button("Refresh page", key='button_refreshpage_code_script_2')
            if button_update_code_script:
                st.experimental_rerun()
            
            with open("libs/script_to_run_out.py", "r") as f:
                file_contents = f.read()

            content = st_ace(file_contents,language='python', theme='pastel_on_dark')

            col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
            with col_code_script_1:
                opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                if opt_code_script == 'Save script':
                    button_code_save_script = st.button("Save", key='button_code_save_script')
                    if button_code_save_script:
                        file = open("libs/script_to_run_out.py", "w") 
                        file.write(content)
                        file.close()
                        time.sleep(1)
                        st.experimental_rerun()
                elif opt_code_script == 'Download script':
                    st.download_button(
                        label="Download script",
                        data=file_contents,
                        file_name='script_to_run_out.py'
                    )
                elif opt_code_script == 'Reset script':
                    button_code_reset_script = st.button("Reset", key='button_code_reset_script_2')
                    if button_code_reset_script:
                        with open("libs/script_to_run_out_backup.py", "r") as f:
                            file_contents_backup = f.read()

                        file = open("libs/script_to_run_out.py", "w") 
                        file.write(file_contents_backup)
                        file.close()
                        time.sleep(1)
                        st.experimental_rerun()                    
        
        with st.expander("Execute script", expanded=False):

            optionDataset = st.selectbox(
            'Select the dataset',
            (st.session_state['dataset']))

            if optionDataset:

                col_apply_code_1, col_apply_code_2 = st.columns([1,4])
                with col_apply_code_1:
                    apply_select = st.selectbox("Apply script to selected dataset?",['No','Yes'])

                submit_script = st.button('Apply', key='submit_script')
                if submit_script:
                    if apply_select == 'Yes':
                        with st.spinner('Wait for it...'):
                            import libs.script_to_run_out as script_run
                            df = script_run.run_out(base=st.session_state[optionDataset])
                            df.execute()
                            st.session_state[optionDataset] = df.base
                            time.sleep(1)
                            st.experimental_rerun()
                    else:
                        st.warning("Choose the option to apply the script!")

            else:

                st.write("There is no Dataset loaded")

        with st.expander("View dataset information",expanded=False):

            #@st.cache
            def func_tab_final():
                if list(st.session_state[optionDataset].select_dtypes(include=[np.number]).columns):
                    tab_numbers = st.session_state[optionDataset].describe(include=[np.number]).T
                else:
                    tab_numbers = pd.DataFrame()
                if list(st.session_state[optionDataset].select_dtypes(include=['object','category']).columns):
                    tab_objects = st.session_state[optionDataset].describe(include=['object','category']).T
                else:
                    tab_objects = pd.DataFrame()

                tab_joins = pd.concat([tab_objects, tab_numbers],axis=1)

                tab_nans = pd.DataFrame(st.session_state[optionDataset].isna().sum(), columns=['NaN'])
                tab_nans['NaN%'] = round((st.session_state[optionDataset].isna().sum() / st.session_state[optionDataset].shape[0])*100,2)

                tab = tab_joins.join(tab_nans)

                dfx_dtypes = pd.DataFrame(st.session_state[optionDataset].dtypes, columns=['type'])
                dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                del dfx_dtypes['type']

                tab_final = tab.join(dfx_dtypes)
                if tab_objects.shape[0] > 0 and tab_numbers.shape[0] > 0:
                    tab_final.columns = ['count', 'unique', 'top', 'freq', 'count_number', 'mean', 'std', 
                        'min', '25%','50%', '75%', 'max', 'NaN', 'NaN%', 'type_str']

                    tab_final['count'] = np.where(tab_final['count'].isnull(), tab_final['count_number'], tab_final['count'])
                    tab_final['count'] = tab_final['count'].astype(int)
                    del tab_final['count_number']
                return tab_final

            col1, col2, col3 = st.columns([0.8,0.8,2.5])
            with col1:
                init_rows = st.number_input("Min. of rows", step=1, value=0, min_value=0)
            with col2:
                final_rows = st.number_input("Max. of rows", step=1, value=st.session_state[optionDataset].shape[0] if st.session_state[optionDataset].shape[0] < 5 else 5, min_value=0, max_value=int(st.session_state[optionDataset].shape[0]))
                
            col11, col22, col33 = st.columns([0.8,0.8,2.5])
            with col11:
                init_cols = st.number_input("Min. of cols", step=1, value=0, min_value=0)
            with col22:
                final_cols = st.number_input("Max. of cols", step=1, value=int(st.session_state[optionDataset].shape[1]), min_value=0, max_value=int(st.session_state[optionDataset].shape[1]))

            selectionColumn = st.checkbox('Columns selections')

            st.header("View dataset")
            if selectionColumn:
                optionsViewColumns = st.multiselect("Multiselect columns",list(st.session_state[optionDataset].columns))
                #st.write(optionsViewColumns)
                st.write(
                    f"""
                    **Columns selected**: {", ".join(optionsViewColumns)}
                    """
                )
                st.dataframe(st.session_state[optionDataset][optionsViewColumns].iloc[int(init_rows):int(final_rows)])
            else:
                st.dataframe(st.session_state[optionDataset].iloc[int(init_rows):int(final_rows), int(init_cols):int(final_cols)])

            st.write("------------------------------------------------------")
            st.header("Descriptive")
            show_me_descriptive = st.checkbox("Show descriptive")
            if show_me_descriptive:
                st.dataframe(func_tab_final())

    def appExecuteScript_in_session():

        st.title('Execute script in session')

        with st.expander("Code in script", expanded=False):

            button_update_code_script_3 = st.button("Refresh page", key='button_refreshpage_code_script_3')
            if button_update_code_script_3:
                st.experimental_rerun()
            
            with open("libs/script_to_run_out_2.py", "r") as f:
                file_contents = f.read()

            content = st_ace(file_contents,language='python', theme='pastel_on_dark')

            col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
            with col_code_script_1:
                opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                if opt_code_script == 'Save script':
                    button_code_save_script = st.button("Save", key='button_code_save_script_2')
                    if button_code_save_script:
                        file = open("libs/script_to_run_out_2.py", "w") 
                        file.write(content)
                        file.close()
                        time.sleep(1)
                        st.experimental_rerun()
                elif opt_code_script == 'Download script':
                    st.download_button(
                        label="Download script",
                        data=file_contents,
                        file_name='script_to_run_out_2.py'
                    )
                elif opt_code_script == 'Reset script':
                    button_code_reset_script = st.button("Reset", key='button_code_reset_script_3')
                    if button_code_reset_script:
                        with open("libs/script_to_run_out_2_backup.py", "r") as f:
                            file_contents_backup = f.read()

                        file = open("libs/script_to_run_out_2.py", "w") 
                        file.write(file_contents_backup)
                        file.close()
                        time.sleep(1)
                        st.experimental_rerun() 

        with st.expander("Execute script", expanded=False): 

            col_apply_code_1, col_apply_code_2 = st.columns([1,4])
            with col_apply_code_1:
                apply_select = st.selectbox("Execute?",['No','Yes'])

            submit_script = st.button('Apply', key='submit_script_2')
            if submit_script:
                if apply_select == 'Yes':
                    with st.spinner('Wait for it...'):
                        import libs.script_to_run_out_2 as script_run_2
                        script_run_2.main()
                        time.sleep(1.5)
                        st.success("Script executed successfully!")
                        time.sleep(1)
                        st.experimental_rerun()
                else:
                    st.warning("Choose the option to execute the script!")

    # -----------------------------------------------------------------------------------------------------------


    if appSelectionSubCat == 'Automated Pre-processing':
        appAutomatedPreProcessing()

    elif appSelectionSubCat == 'Execute Script in Object':
        appExecuteScript()

    elif appSelectionSubCat == 'Execute Script in Session':
        appExecuteScript_in_session()

    elif appSelectionSubCat == 'Home':

        #st.image(Image.open('images/image8.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Data Preparation
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )


        

        



