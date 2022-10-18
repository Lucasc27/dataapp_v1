import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time
import sys
import libs.feature_engineering as FE
import phik
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_option_menu import option_menu

def app():

    appSelectionSubCat = option_menu('Select option', ['Home','Feature Combinations','Feature Selection'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    #appSelectionSubCat = st.sidebar.selectbox('Submenu',['Home','Feature Combinations'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appFeatureCombination():

        st.title('Feature Combinations')
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            with st.expander("Settings", expanded=False):

                varsBlackList = None
                varsWhiteList = None
                len_list_varsBlackList = 0
                len_list_varsWhiteList = 0
                n_cols = st.session_state[optionDataset].shape[1]


                # Black List -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.5,3])
                with col1:
                    has_bl = st.selectbox("Has blacklist", ['No', 'Yes'])
                with col2:
                    if has_bl == 'Yes':
                        varsBlackList = st.multiselect('Input the black list variables', list(st.session_state[optionDataset].columns))
                        len_list_varsBlackList = len(varsBlackList)

                # White List -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.5,3])
                with col1:
                    has_wl = st.selectbox("Has whitelist", ['No', 'Yes'])
                with col2:
                    if has_wl == 'Yes':
                        varsWhiteList = st.multiselect('Input the white list variables', list(st.session_state[optionDataset].columns))
                        len_list_varsWhiteList = len(varsWhiteList)

                #Group List -----------------------------------------------------------------------------------
                col1, col2, col3, col4 = st.columns([0.7,1,2.5,0.5])
                name_group = ''
                cols_of_group = None
                with col1:
                    has_groups = st.selectbox("Has groups", ['No', 'Yes'])
                with col2:
                    if has_groups == 'Yes':
                        name_group = st.selectbox('Input group name', ['group'+str(n) for n in range(1,11)])
                with col3:
                    if name_group != '':
                        cols_of_group = st.multiselect("Input columns in group", st.session_state[optionDataset].columns)
                with col4:
                    if cols_of_group:
                        st.write(" ")
                        st.write(" ")
                        button_add_group = st.button("Add group", key='button_add_group')
                        if button_add_group:
                            if not 'dic_group_feature_combination' in st.session_state['Variables']:
                                st.session_state['Variables'].append('dic_group_feature_combination')
                                st.session_state['dic_group_feature_combination'] = {}
                                st.session_state['dic_group_feature_combination'][name_group] = cols_of_group
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                st.session_state['dic_group_feature_combination'][name_group] = cols_of_group
                                time.sleep(1)
                                st.experimental_rerun()
                # -----------------------------------------------------------------------------------

                col1, col2, col3 = st.columns(3)
                with col1:
                    targetName = st.selectbox("Input the target variable", list(st.session_state[optionDataset].columns.insert(0,None)))
                    len_targetName = 0 if targetName == None else 1

                if not 'dic_group_feature_combination' in st.session_state['Variables']:
                    value_max_coluns_combinations = n_cols - len_list_varsBlackList - len_targetName
                else:
                    count_vars_in_group = 0
                    count_group = 0
                    for group in st.session_state['dic_group_feature_combination']:
                        count_group += 1
                        for col in st.session_state['dic_group_feature_combination'][group]:
                            count_vars_in_group += 1
                    #count_group, count_vars_in_group
                    value_max_coluns_combinations = (n_cols - len_list_varsBlackList - len_targetName - count_vars_in_group)+count_group

                with col2:
                    n_min = st.number_input('Input the minimum number of combinations', step=1, min_value=2)

                with col3:
                    n_max = st.number_input('Input the maximum number of combinations', step=1, value=value_max_coluns_combinations, max_value=value_max_coluns_combinations)

                col1, col2 = st.columns([1.53,3])
                with col1:
                    ObjectOptionsToSave = st.text_input("Input variable name")

                #click_teste = st.form_submit_button("Submit")
                click_teste = st.button("Submit")

                if click_teste:
                    if ObjectOptionsToSave:
                        if ObjectOptionsToSave not in st.session_state:
                            if targetName != None:

                                features = FE.featureCombinations(
                                        base=st.session_state[optionDataset],
                                        target=targetName,
                                        n_min = n_min,
                                        n_max = n_max,
                                        black_list = varsBlackList,
                                        white_list = varsWhiteList,
                                        list_groups = st.session_state['dic_group_feature_combination'] if 'dic_group_feature_combination' in st.session_state['Variables'] else None
                                )
                                verify_allow_combnations = None
                                dic_combinations, verify_allow_combnations = features.toCombine()

                                if verify_allow_combnations:
                                    st.session_state["Objects"].append(ObjectOptionsToSave)
                                    st.session_state["Variables"].append(ObjectOptionsToSave)
                                    st.session_state[ObjectOptionsToSave] = dic_combinations
                                #st.markdown("**Combinations**")
                                #st.write(features.batch_combinations)
                                #st.markdown("**Total of combinations**")
                                #st.write(features.total_combination)

                                    df_comb = pd.DataFrame(st.session_state[ObjectOptionsToSave])
                                    df_comb.to_csv(f'files_output/combinations_{ObjectOptionsToSave}.csv', index=False)

                                    st.success('Combination created successfully')
                                else:
                                    st.warning("Erro...")
                                time.sleep(2)
                                st.experimental_rerun()

                            else:
                                st.error('You need put the target variable name')
                                time.sleep(0.1)
                                st.experimental_rerun()
                            
                        else:
                            st.error('This variable exists in the system')
                            time.sleep(0.1)
                            st.experimental_rerun()
                    else:
                        st.error('Input the variable name')
                        time.sleep(0.1)
                        st.experimental_rerun()

            with st.expander("Groups", expanded=False):

                col1, col2, col3, col4 = st.columns([0.7,1,2.5,0.5])
                name_group = ''
                cols_of_group = None
                select_to_delete_group = None
                with col1:
                    create_or_delete_group = st.selectbox("Create or delete groups", ['None','Create', 'Delete'])
                with col2:
                    if create_or_delete_group == 'Create':
                        name_group = st.selectbox('Input group name', ['group'+str(n) for n in range(1,11)])
                    elif create_or_delete_group == 'Delete':
                        if 'dic_group_feature_combination' in st.session_state['Variables']:
                            select_to_delete_group = st.selectbox("Select group to delete", st.session_state['dic_group_feature_combination'])
                        else:
                            st.warning("Don't have group of variables")
                with col3:
                    if name_group != '':
                        cols_of_group = st.multiselect("Input columns in group", st.session_state[optionDataset].columns)
                    elif select_to_delete_group:
                        st.write(" ")
                        st.write(" ")
                        button_to_delete_group = st.button("Delete group")
                        if button_to_delete_group:
                            del st.session_state['dic_group_feature_combination'][select_to_delete_group]
                            time.sleep(1)
                            st.experimental_rerun()
                with col4:
                    if cols_of_group:
                        st.write(" ")
                        st.write(" ")
                        button_add_group = st.button("Add group", key='button_add_group')
                        if button_add_group:
                            if not 'dic_group_feature_combination' in st.session_state['Variables']:
                                st.session_state['Variables'].append('dic_group_feature_combination')
                                st.session_state['dic_group_feature_combination'] = {}
                                st.session_state['dic_group_feature_combination'][name_group] = cols_of_group
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                st.session_state['dic_group_feature_combination'][name_group] = cols_of_group
                                time.sleep(1)
                                st.experimental_rerun()

            with st.expander("View Variables", expanded=False):
                
                if st.session_state['Variables']:
                    
                    SelectionViewVariable = st.selectbox(
                    'Select a variable', (st.session_state['Variables']))

                    # Contagem de combinações batch e total
                    if SelectionViewVariable != 'dic_group_feature_combination':
                        lista_contagem = []
                        for x in range(0, len(st.session_state[SelectionViewVariable])):
                            lista_contagem.append(len(st.session_state[SelectionViewVariable][x]))
                            
                        total_combination = len(st.session_state[SelectionViewVariable])
                        batch_combinations = pd.DataFrame(lista_contagem, columns=['Qnt_var']).value_counts()
                        batch_combinations = pd.DataFrame(batch_combinations, columns=['Total'])

                        showbase = st.checkbox('Preview')
                        if showbase:
                            st.write("Total combinations", total_combination)
                            st.write("Features by combination", batch_combinations)
                            st.write("List of variables", st.session_state[SelectionViewVariable])

                    elif SelectionViewVariable == 'dic_group_feature_combination':
                        st.write(st.session_state['dic_group_feature_combination'])
                else:
                    st.write("There is not variables loaded")

            #with st.expander("Loading combinations", expanded=False):

            #    dataset_vars = list(st.session_state['dataset'])
            #    objects_vars = list(st.session_state['Objects'])
            #    objs_final = [objects_vars for objects_vars in objects_vars if objects_vars not in dataset_vars]

            #    NameVariable = st.selectbox(
            #    'Select the variable to save',
            #    (objs_final))

            #    if NameVariable:

            #            myBase = st.radio("Select an option",('Upload csv','Upload xlsx'))
            #            with st.form(key='my_form_load_combinations', clear_on_submit=True):
            #                if not NameVariable in st.session_state['Variables']:
            #                    if myBase == 'Upload csv':

            #                        check_delimiter = st.checkbox("Custom Delimiter")
            #                        deLim = st.text_input('Input the delimiter')
            #                        
            #                        if check_delimiter:
            #                            uploaded_file = st.file_uploader("Choose a file")
            #                            if uploaded_file is not None:

            #                                bytes_data = uploaded_file.getvalue()

            #                                df_default = pd.read_csv(uploaded_file, sep=str(deLim))

            #                                combs_list = []
            #                                for i in range(0,len(df_default)):
            #                                    item = df_default.iloc[i,:].tolist()
            #                                    item = [item for item in item if str(item) != 'nan']
            #                                    combs_list.append(item)

            #                                st.session_state["Variables"].append(NameVariable)
            #                                st.session_state[NameVariable] = combs_list
            #                                #st.session_state['have_dataset'] = True

            #                        else:
            #                            uploaded_file = st.file_uploader("Choose a file")
            #                            if uploaded_file is not None:

            #                                bytes_data = uploaded_file.getvalue()

            #                                df_default = pd.read_csv(uploaded_file)

            #                                combs_list = []
            #                                for i in range(0,len(df_default)):
            #                                    item = df_default.iloc[i,:].tolist()
            #                                    item = [item for item in item if str(item) != 'nan']
            #                                    combs_list.append(item)

            #                                st.session_state["Variables"].append(NameVariable)
            #                                st.session_state[NameVariable] = combs_list
            #                                #st.session_state['have_dataset'] = True

            #                    elif myBase == 'Upload xlsx':
            #                            
            #                        uploaded_file = st.file_uploader("Choose a file")
            #                        if uploaded_file is not None:

            #                            bytes_data = uploaded_file.getvalue()

            #                            df_default = pd.read_excel(uploaded_file)

            #                            combs_list = []
            #                            for i in range(0,len(df_default)):
            #                                item = df_default.iloc[i,:].tolist()
            #                                item = [item for item in item if str(item) != 'nan']
            #                                combs_list.append(item)

            #                            st.session_state["Variables"].append(NameVariable)
            #                            st.session_state[NameVariable] = combs_list
            #                            #st.session_state['have_dataset'] = True

            #                else:
            #                    st.write("<u><b>Variable loaded</b></u>", unsafe_allow_html=True)

            #                submit_var_comb_load = st.form_submit_button('Submit')
            #                if submit_var_comb_load:
            #                    #time.sleep(0.2)
            #                    st.experimental_rerun()


        else:

            st.write("There is no Dataset loaded")

    def appFeatureSelection():

        FeatureSelectionType = option_menu('', ['VIF, WoE and IV (Multicollinearity)','Features Importance'], 
        default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

        if FeatureSelectionType == 'VIF, WoE and IV (Multicollinearity)':

            with st.expander("Multicollinearity", expanded=False):
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    var_dataset_vif = st.selectbox('Select the dataset',(st.session_state['dataset']))
                with col2:
                    var_target_vif = st.selectbox("Select the target", st.session_state[var_dataset_vif].columns)

                col1, col2, col3 = st.columns([0.2,0.8,0.5])
                with col1:
                    to_remov_var_vif = st.selectbox("Remove variable?",['No','Yes'])
                with col2:
                    if to_remov_var_vif == 'Yes':
                        remov_var_vif = st.multiselect("Select the variables to remove", st.session_state[var_dataset_vif].columns)

                col1, col2, col3 = st.columns([0.09,0.15,0.5])
                with col1:
                    type_correlation = st.selectbox("Correlation type",['Pearson','Spearman','Phik'])
                with col2:
                    number_of_bins = st.number_input("Input the binning number for WoE", min_value=2, step=1, value=2)

                vif_button = st.button("Apply", key='vif_button')
                if vif_button:
                    with st.spinner('Wait for it...'):
                        from statsmodels.stats.outliers_influence import variance_inflation_factor
                        if to_remov_var_vif == 'Yes':
                            var_remov_del = [var_target_vif] + remov_var_vif
                            X = st.session_state[var_dataset_vif].drop(var_remov_del,axis=1)
                        else:
                            X = st.session_state[var_dataset_vif].drop([var_target_vif],axis=1)
                        
                        # VIF dataframe
                        vif_data = pd.DataFrame()
                        vif_data["feature"] = X.columns
                        
                        # calculating VIF for each feature
                        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                                                for i in range(len(X.columns))]
                        st.subheader("Variance Inflation Factor Table")
                        st.dataframe(vif_data)

                        if type_correlation == 'Pearson':
                            corr = X.corr(method = 'pearson')
                        elif type_correlation == 'Spearman':
                            corr = X.corr(method = 'spearman')
                        elif type_correlation == 'Phik':
                            corr = phik.phik_matrix(X)

                        st.subheader("Correlation table")
                        st.dataframe(corr)

                        def iv_woe(data, target, bins=10, show_woe=False):
                            #Empty Dataframe
                            newDF,woeDF = pd.DataFrame(), pd.DataFrame()
                            
                            #Extract Column Names
                            cols = data.columns
                            
                            #Run WOE and IV on all the independent variables
                            for ivars in cols[~cols.isin([target])]:
                                if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
                                    binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
                                    d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                                else:
                                    d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

                                
                                # Calculate the number of events in each group (bin)
                                d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
                                d.columns = ['Cutoff', 'N', 'Events']
                                d['Cutoff'] = d['Cutoff'].astype(str)
                                
                                # Calculate % of events in each group.
                                d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

                                # Calculate the non events in each group.
                                d['Non-Events'] = d['N'] - d['Events']
                                # Calculate % of non events in each group.
                                d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

                                # Calculate WOE by taking natural log of division of % of non-events and % of events
                                d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
                                d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
                                d.insert(loc=0, column='Variable', value=ivars)
                                print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
                                temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
                                newDF=pd.concat([newDF,temp], axis=0)
                                woeDF=pd.concat([woeDF,d], axis=0)

                                #Show WOE Table
                                if show_woe == True:
                                    print(d)
                            return newDF, woeDF

                        df_IV, df_All = iv_woe(st.session_state[var_dataset_vif], var_target_vif, bins=number_of_bins, show_woe=False)

                        col1, col2 = st.columns([0.25,0.75])
                        with col1:
                            st.subheader("Information value table")
                            st.dataframe(df_IV)
                        with col2:
                            st.subheader("WoE and IV calculation table")
                            st.dataframe(df_All)
                        st.image(Image.open('images/table_iv.png'), width=500)

                        df_corr = pd.DataFrame(columns=['Variable'])
                        x = 0
                        for i in range(0, len(corr)):
                            for col in corr.columns:
                                df_corr.loc[x,'Variable'] = corr.iloc[i:i+1].index[0]
                                df_corr.loc[x,'Variable(b)'] = col
                                df_corr.loc[x,'Coeff-correlation'] = corr[col][i]
                                x += 1
                                
                        df_corr_IV = df_corr.merge(df_IV, how='left', on='Variable')
                        df_corr_IV.columns = ['Variable(a)','Variable','Coeff-correlation','IV_variable_a']

                        df_corr_IV = df_corr_IV.merge(df_IV, how='left', on='Variable')
                        df_corr_IV.columns = ['Variable(a)','Variable(b)','Coeff-correlation','IV_variable_a','IV_variable_b']

                        df_corr_IV = df_corr_IV[df_corr_IV['Variable(a)'] != df_corr_IV['Variable(b)']]
                        df_corr_IV.reset_index(inplace=True, drop=True)

                        st.subheader("Table of correlation coefficients with the information value")
                        st.dataframe(df_corr_IV)

        elif FeatureSelectionType == 'Features Importance':

            with st.expander("Features Importance with tree models", expanded=False):

                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    var_dataset_FiTree = st.selectbox('Select the dataset',(st.session_state['dataset']))
                with col2:
                    var_target_FiTree = st.selectbox("Select the target", st.session_state[var_dataset_FiTree].columns)

                col1, col2, col3 = st.columns([0.2,0.8,0.5])
                with col1:
                    to_remov_var_FiTree = st.selectbox("Remove variable?",['No','Yes'])
                with col2:
                    if to_remov_var_FiTree == 'Yes':
                        remov_var_FiTree = st.multiselect("Select the variables to remove", st.session_state[var_dataset_FiTree].columns)

                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    list_models_FiTree = ['XGBClassifier','ExtraTreesClassifier','GradientBoostingClassifier','AdaBoostClassifier','LGBMClassifier','DecisionTreeClassifier','RandomForestClassifier']
                    var_models_FiTree = st.multiselect('Select the model(s)',list_models_FiTree)
                with col2:
                    shap_or_importanceTree = st.selectbox("Use features importance from tree models or shap values", ['Tree-based','Shap value'])
                with col3:
                    if shap_or_importanceTree == 'Tree-based':
                        qtd_features = st.number_input("Total number of features", value=len(st.session_state[var_dataset_FiTree].drop([var_target_FiTree],axis=1).columns.tolist()), step=1, min_value=1)
                    else:
                        max_display_shap = st.number_input("Select max display on the chart", value=20 if len(st.session_state[var_dataset_FiTree].drop([var_target_FiTree],axis=1).columns.tolist()) > 20 else len(st.session_state[var_dataset_FiTree].drop([var_target_FiTree],axis=1).columns.tolist()), step=1, min_value=1, max_value=len(st.session_state[var_dataset_FiTree].drop([var_target_FiTree],axis=1).columns.tolist()))

                FiTree_button = st.button("Apply", key='FiTree_button')
                if FiTree_button:
                    with st.spinner('Wait for it...'):
                        if var_models_FiTree:

                            FiTree_models = []
                            if 'XGBClassifier' in var_models_FiTree:
                                from xgboost import XGBClassifier
                                FiTree_models.append(XGBClassifier(random_state=42))
                            if 'ExtraTreesClassifier' in var_models_FiTree:
                                from sklearn.ensemble import ExtraTreesClassifier
                                FiTree_models.append(ExtraTreesClassifier(random_state=42))
                            #if 'BaggingClassifier' in var_models_FiTree:
                            #    from sklearn.ensemble import BaggingClassifier
                            #    FiTree_models.append(BaggingClassifier(random_state=42))
                            if 'GradientBoostingClassifier' in var_models_FiTree:
                                from sklearn.ensemble import GradientBoostingClassifier
                                FiTree_models.append(GradientBoostingClassifier(random_state=42))
                            if 'AdaBoostClassifier' in var_models_FiTree:
                                from sklearn.ensemble import AdaBoostClassifier
                                FiTree_models.append(AdaBoostClassifier(random_state=42))
                            if 'LGBMClassifier' in var_models_FiTree:
                                from lightgbm import LGBMClassifier
                                FiTree_models.append(LGBMClassifier(random_state=42))
                            if 'DecisionTreeClassifier' in var_models_FiTree:
                                from sklearn.tree import DecisionTreeClassifier
                                FiTree_models.append(DecisionTreeClassifier(random_state=42))
                            if 'RandomForestClassifier' in var_models_FiTree:
                                from sklearn.ensemble import RandomForestClassifier
                                FiTree_models.append(RandomForestClassifier(random_state=42))

                            if to_remov_var_FiTree == 'Yes':
                                var_remov_FiTree = [var_target_FiTree] + remov_var_FiTree
                                X = st.session_state[var_dataset_FiTree].drop(var_remov_FiTree,axis=1)
                                y = st.session_state[var_dataset_FiTree][var_target_FiTree]
                            else:
                                X = st.session_state[var_dataset_FiTree].drop([var_target_FiTree],axis=1)
                                y = st.session_state[var_dataset_FiTree][var_target_FiTree]

                            if shap_or_importanceTree == "Tree-based":

                                for (model, name_model) in zip(FiTree_models, var_models_FiTree):

                                    model.fit(X, y)
                                    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])
                                    col1, col2 = st.columns([1.5,0.5])
                                    with col1:
                                        fig, ax = plt.subplots()
                                        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:int(qtd_features)], ax=ax)
                                        #plt.title('Important Features')
                                        st.pyplot(fig)
                                        #fig = px.bar(feature_imp.sort_values(by="Value", ascending=False).iloc[:int(qtd_features)], x='Value', y='Feature', color='Value', orientation='h', title=name_model)
                                        #st.plotly_chart(fig, use_container_width=True)

                            else:

                                import shap
                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

                                for (model, name_model) in zip(FiTree_models, var_models_FiTree):

                                    model.fit(X_train, y_train)

                                    shap.initjs()
                                    explainer = shap.TreeExplainer(model, link="logit")
                                    shap_values = explainer.shap_values(X_test)
                                    col1, col2 = st.columns([1.5,0.5])
                                    with col1:
                                        st.subheader(name_model)
                                        summary_plot = shap.summary_plot(shap_values[1], X_test, plot_type="bar", max_display=int(max_display_shap))
                                        st.pyplot(summary_plot)


                    

                        else:
                            st.warning("Select the model(s)!")


    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Feature Combinations':
        appFeatureCombination()

    elif appSelectionSubCat == 'Feature Selection':
        appFeatureSelection()

    elif appSelectionSubCat == 'Home':

        #st.image(Image.open('images/feature_engineering.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Feature Engeneering
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )

        