
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import mlflow
import libs.EDA_graphs as EDA
import json
import time
import libs.model_engineering_new as me
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
import pickle
import os
st.set_option('deprecation.showPyplotGlobalUse', False)


def app():

    appSelectionSubCat = option_menu('Select option', ['Home','Models Experiments', 'Report Experiments','Hyperparameter Tuning','Model validation','Save models','Production'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    #appSelectionSubCat = st.sidebar.selectbox('Submenu',['Home','Models Experiments', 'Report Experiments'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appModelsExperiments():

        st.title('Models experiments')

        TypeMachineLearningModelsExperiments = option_menu('', ['Supervised Learning','Unsupervised Learning'], 
                default_index=0, orientation="horizontal",
                styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                        "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
            )

        if TypeMachineLearningModelsExperiments == 'Supervised Learning':

            with st.expander("Experiments", expanded=False):

                list_to_select_models = ["LogisticRegression", "XGBClassifier", "LGBMClassifier", "RandomForestClassifier", 
                "AdaBoostClassifier", "GradientBoostingClassifier", "LinearDiscriminantAnalysis", "GaussianNB", 
                "DecisionTreeClassifier", "KNeighborsClassifier"]

                targetName = None

                col1, col2, col3 = st.columns([1,1,3])
                with col1:
                    select_or_input_experiments = st.selectbox("Select experiment or create new", ['None','Select experiment','New experiment'])
                with col2:
                    if select_or_input_experiments == 'Select experiment':
                        experiments = mlflow.list_experiments()
                        list_name_of_experiments = [experiments.name for experiments in experiments]
                        list_name_of_experiments.insert(0, 'None')
                        nameExperiment = st.selectbox("Select experiment", list_name_of_experiments)
                    elif select_or_input_experiments == 'New experiment':
                        nameExperiment = st.text_input("Input the name of experiment")

                # -----------------------------------------------------------------------------------
                if select_or_input_experiments != 'None':
                    col1, col2, col3, col4 = st.columns([1,1,1,1])
                    with col1:
                        select_X_train = st.selectbox("Select (X) train", st.session_state['dataset'])
                    with col2:
                        select_y_train = st.selectbox("Select (y) train", st.session_state['dataset'])
                    with col3:
                        select_X_test = st.selectbox("Select (X) test", st.session_state['dataset'])
                    with col4:
                        select_y_test = st.selectbox("Select (y) test", st.session_state['dataset'])

                    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                    with col1:
                        targetName = st.selectbox("Input the target name", list(st.session_state[select_y_train].columns.insert(0,'None')))

                # -----------------------------------------------------------------------------------
                
                if targetName != 'None' and targetName:
                    col1, col2 = st.columns([1,4])
                    with col1:
                        valid_features = st.selectbox("Combination or select features", ['None','Select features', 'Combination features'])
                    with col2:
                        if valid_features == 'Select features':
                            combinations_features_pre = st.multiselect('Select the columns', st.session_state[select_X_train].columns)
                            combinations_features = []
                            combinations_features.append(list(combinations_features_pre))
                        elif valid_features == 'Combination features':
                            if len(st.session_state["Variables"]) != 0:
                                list_combination_features = st.selectbox('Select the variable', list(st.session_state["Variables"]))
                                combinations_features = st.session_state[list_combination_features]
                            else:
                                st.warning("There is not feature combination variable")

                # -----------------------------------------------------------------------------------
                if targetName != 'None' and targetName:
                    col1, col2 = st.columns([0.5,3])
                    with col1:
                        select_or_input_models = st.selectbox("Select models type", ['None','Select models', 'Input models'])
                    with col2:
                        if select_or_input_models == 'Select models':
                            list_models_runs = st.multiselect('Select the models', list_to_select_models)
                        elif select_or_input_models == 'Input models':
                            st.error("construction....")
                # -----------------------------------------------------------------------------------

                    #shap_output = st.checkbox("Interpretation with shape values")

                    click_run = st.button("Submit", key='click_run_experiments')

                    if click_run:

                        if valid_features == 'None':
                            combinations_features = None

                        if not nameExperiment:
                            nameExperiment = 'MLFLOW MODELS TESTING'

                        with st.spinner('Wait for it...'):
                            X_train = st.session_state[select_X_train]
                            y_train = st.session_state[select_y_train]

                            X_test = st.session_state[select_X_test]
                            y_test = st.session_state[select_y_test]

                            run_models = me.modelExperimentSetting(
                                have_base=None, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                target = targetName, cols_combinations = combinations_features, models = list_models_runs, 
                                mlflow_name_experiment = nameExperiment)
                                
                            run_models.execute()

        elif TypeMachineLearningModelsExperiments == 'Unsupervised Learning':
            
            TypeClusteringModel = option_menu('', ['KMeans','DBSCAN','MeanShift','AgglomerativeClustering'], 
                default_index=0, orientation="horizontal",
                styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                        "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
            )

            if TypeClusteringModel == 'KMeans':

                with st.expander("Information", expanded=False):
                    info_silhouette = st.checkbox("?Info silhouette score", key='info_silhouette')
                    info_elbow = st.checkbox("?Info method elbow", key='info_silhouette')
                    if info_silhouette:
                        st.info("Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters.")
                        st.image(Image.open('images/silhouette_distance.png'), width=600)
                    if info_elbow:
                        st.info("The elbow method runs k-means clustering on the dataset for a range of values for k (say from 1-10) and then for each value of k computes an average score for all clusters. By default, the distortion score is computed, the sum of square distances from each point to its assigned center.")
                        st.image(Image.open('images/elbow_method.png'), width=600)
                    
                with st.expander("Cluster Metrics", expanded=False):

                    combinations_features = None
                    remove_vars_combinations = None

                    col1, col2, col3, col4 = st.columns([1,0.6,0.4,1])
                    with col1:
                        var_dataset_metrics = st.selectbox("Select the dataset", st.session_state['dataset'], key='var_dataset_metrics')
                    with col2:
                        valid_features_metrics = st.selectbox("Select option to apply",['None','Select variables','Remove variables','Combination features'], key='valid_features_metrics')
                    with col3:
                        go_scaler = st.selectbox("Use StandardScaler?",['Yes', 'No'])

                    if valid_features_metrics == 'Select variables':
                            col1, col2 = st.columns([1.35,0.66])
                            with col1:
                                select_features = st.multiselect('Select the columns', st.session_state[var_dataset_metrics].columns)
                    elif valid_features_metrics == 'Combination features':
                        col1, col2, col3, col4 = st.columns([0.6,0.6,0.8,1])
                        with col1:
                            if len(st.session_state["Variables"]) != 0:
                                list_combination_features = st.selectbox('Select the variable', list(st.session_state["Variables"]))
                                combinations_features = st.session_state[list_combination_features]
                            else:
                                st.warning("There is not feature combination variable")
                        with col2:
                            if len(st.session_state["Variables"]) != 0:
                                remove_vars_combinations = st.selectbox("Remove variables of combinations?", ['No','Yes'])
                        with col3:
                            if remove_vars_combinations == 'Yes':
                                list_remove_vars_combinations = st.multiselect("Select the variables to remove of combinations", st.session_state[var_dataset_metrics].columns, key='list_remove_vars_combinations')
                    elif valid_features_metrics == 'Remove variables':
                        col1, col2 = st.columns([1.35,0.66])
                        with col1:
                            remov_var_metrics = st.multiselect("Select the variables to remove", st.session_state[var_dataset_metrics].columns, key='remov_var_metrics')

                    col1, col2, col3 = st.columns([0.09,0.16,0.5])
                    with col1:
                        select_type_k = st.selectbox("Select a list or range of K", ['Range','List'])
                    with col2:
                        if select_type_k == 'Range':
                            number_of_k_metrics = st.number_input("Input the k value", min_value=3, step=1, value=3, key='number_of_k_metrics')
                        else:
                            number_of_k_metrics = st.text_input("Input values separated by comma, ex: 3,4,5")
                            if number_of_k_metrics != '':
                                number_of_k_metrics = [int(num) for num in number_of_k_metrics.split(',')]

                    col1, col2, col3 = st.columns([0.5,0.5,0.5])
                    with col1:
                        show_metrics = st.multiselect("Show me metrics", ['Elbow','Silhouette','Intercluster','Tab Count','PCA'])

                    button_metrics = st.button("Apply", key='button_metrics')
                    st.write("--------------------------")
                    if button_metrics:

                        from sklearn.preprocessing import StandardScaler

                        if valid_features_metrics == 'None':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_metrics])
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_metrics].columns)
                            else:
                                X =  st.session_state[var_dataset_metrics]
                        elif valid_features_metrics == 'Select variables':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_metrics][select_features])
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_metrics][select_features].columns)
                            else:
                                X = st.session_state[var_dataset_metrics][select_features]
                        elif valid_features_metrics == 'Combination features':
                            if go_scaler == 'Yes':
                                if remove_vars_combinations == 'Yes':
                                    X = StandardScaler().fit_transform(st.session_state[var_dataset_metrics].drop(list_remove_vars_combinations,axis=1))
                                    X = pd.DataFrame(X, columns=st.session_state[var_dataset_metrics].drop(list_remove_vars_combinations,axis=1).columns)
                                else:
                                    X = StandardScaler().fit_transform(st.session_state[var_dataset_metrics])
                                    X = pd.DataFrame(X, columns=st.session_state[var_dataset_metrics].columns)
                            else:
                                if remove_vars_combinations == 'Yes':
                                    X = st.session_state[var_dataset_metrics].drop(list_remove_vars_combinations,axis=1)
                                else:
                                    X = st.session_state[var_dataset_metrics]
                        elif valid_features_metrics == 'Remove variables':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_metrics].drop(remov_var_metrics,axis=1))
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_metrics].drop(remov_var_metrics,axis=1).columns)
                            else:
                                X = st.session_state[var_dataset_metrics].drop(remov_var_metrics,axis=1)

                        from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance
                        from sklearn.metrics import silhouette_samples, silhouette_score
                        from sklearn.decomposition import PCA
                        from yellowbrick.cluster.elbow import kelbow_visualizer
                        from sklearn.cluster import KMeans

                        if combinations_features:

                            for combination in combinations_features:

                                st.subheader(f"Combination {combination}")
                                col1, col2 = st.columns([1.2,0.8])
                                with col1:
                                    if 'Elbow' in show_metrics:
                                        fig, ax = plt.subplots()
                                        kelbow_visualizer(KMeans(random_state=42), X[combination], k=(2,number_of_k_metrics+1) if select_type_k == 'Range' else number_of_k_metrics, ax=ax)
                                        st.pyplot(fig)

                                for k in range(2, number_of_k_metrics+1) if select_type_k == 'Range' else number_of_k_metrics:
                                    model = KMeans(k, random_state=42)
                                    cluster_labels = model.fit_predict(X[combination])

                                    if 'Silhouette' in show_metrics:
                                        var_silhouette_score = silhouette_score(X[combination], cluster_labels)
                                        st.subheader(f"**The mean silhouette score for k = {k} is {round(var_silhouette_score,4)}**")

                                    col1, col2, col3 = st.columns([1.2,1.2,0.6])
                                    with col1:
                                        if 'Silhouette' in show_metrics:
                                            fig, ax = plt.subplots()
                                            viz_Silhouette = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
                                            viz_Silhouette.fit(X[combination])
                                            viz_Silhouette.poof()
                                            st.pyplot(fig)
                                    with col2:
                                        if 'Intercluster' in show_metrics:
                                            st.write("  ")
                                            fig, ax = plt.subplots()
                                            viz_Intercluster = InterclusterDistance(model)
                                            viz_Intercluster.fit(X[combination])        # Fit the data to the visualizer
                                            st.pyplot(fig)
                                    with col3:
                                        if 'Tab Count' in show_metrics:
                                            st.write(" ")
                                            table_count_labels = pd.DataFrame(cluster_labels, columns=['labels'])
                                            table_count_labels = pd.DataFrame(table_count_labels.value_counts(), columns=['n_total'])
                                            table_count_labels.reset_index(inplace=True)
                                            st.dataframe(table_count_labels)

                                    if 'PCA' in show_metrics:
                                        pca = PCA(n_components=2)
                                        components = pca.fit_transform(X[combination])

                                        fig = px.scatter(components, x=0, y=1, color=cluster_labels,
                                        title=f"PCA1({round((pca.explained_variance_ratio_[0]*100),2)}% variance) and PCA2({round((pca.explained_variance_ratio_[1]*100),2)}% variance)",
                                        labels={
                                            "0": "PCA1",
                                            "1": "PCA2"
                                        })
                                        st.plotly_chart(fig, use_container_width=True)

                                    if 'Silhouette' in show_metrics or 'Intercluster' in show_metrics or 'Tab Count' in show_metrics or 'PCA' in show_metrics:
                                        st.write("--------------------------")


                        else:

                            st.subheader(f"Features {list(X.columns)}")
                            col1, col2 = st.columns([0.9,1.1])
                            with col1:
                                if 'Elbow' in show_metrics:
                                    fig, ax = plt.subplots()
                                    kelbow_visualizer(KMeans(random_state=42), X, k=(2,number_of_k_metrics+1) if select_type_k == 'Range' else number_of_k_metrics, ax=ax)
                                    st.pyplot(fig)

                            for k in range(2, number_of_k_metrics+1) if select_type_k == 'Range' else number_of_k_metrics:
                                model = KMeans(k, random_state=42)
                                cluster_labels = model.fit_predict(X)

                                if 'Silhouette' in show_metrics:
                                    var_silhouette_score = silhouette_score(X, cluster_labels)
                                    st.subheader(f"**The mean silhouette score for k = {k} is {round(var_silhouette_score,4)}**")

                                st.header(f"K = {k}")

                                col1, col2, col3 = st.columns([1.2,1.2,0.6])
                                with col1:
                                    if 'Silhouette' in show_metrics:
                                        fig, ax = plt.subplots()
                                        viz_Silhouette = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
                                        viz_Silhouette.fit(X)
                                        viz_Silhouette.poof()
                                        st.pyplot(fig)
                                with col2:
                                    if 'Intercluster' in show_metrics:
                                        st.write(" ")
                                        fig, ax = plt.subplots()
                                        viz_Intercluster = InterclusterDistance(model, random_state=42)
                                        viz_Intercluster.fit(X)        # Fit the data to the visualizer
                                        st.pyplot(fig)
                                with col3:
                                    if 'Tab Count' in show_metrics:
                                        st.write(" ")
                                        table_count_labels = pd.DataFrame(cluster_labels, columns=['labels'])
                                        table_count_labels = pd.DataFrame(table_count_labels.value_counts(), columns=['n_total'])
                                        table_count_labels.reset_index(inplace=True)
                                        st.dataframe(table_count_labels)

                                if 'PCA' in show_metrics:
                                    pca = PCA(n_components=2,random_state=42)
                                    components = pca.fit_transform(X)

                                    fig = px.scatter(components, x=0, y=1, color=cluster_labels,
                                    title=f"PCA1({round((pca.explained_variance_ratio_[0]*100),2)}% variance) and PCA2({round((pca.explained_variance_ratio_[1]*100),2)}% variance)",
                                    labels={
                                        "0": "PCA1",
                                        "1": "PCA2"
                                    })
                                    st.plotly_chart(fig, use_container_width=True)

                                if 'Silhouette' in show_metrics or 'Intercluster' in show_metrics or 'Tab Count' in show_metrics or 'PCA' in show_metrics:
                                    st.write("--------------------------")

                with st.expander("Experiment", expanded=False):

                    combinations_features = None
                    remove_vars_combinations = None

                    col1, col2, col3, col4 = st.columns([1,0.6,0.4,1])
                    with col1:
                        var_dataset_cluster = st.selectbox("Select the dataset", st.session_state['dataset'], key='var_dataset_cluster')
                    with col2:
                        valid_features_cluster = st.selectbox("Select option to apply",['None','Select variables','Remove variables','Combination features'], key='valid_features_cluster')
                    with col3:
                        go_scaler = st.selectbox("Use StandardScaler?",['Yes', 'No'], key='go_scaler_cluster')

                    if valid_features_cluster == 'Select variables':
                            col1, col2 = st.columns([1.35,0.66])
                            with col1:
                                select_features_cluster = st.multiselect('Select the columns', st.session_state[var_dataset_cluster].columns, key='select_features_cluster')
                    elif valid_features_cluster == 'Combination features':
                        col1, col2, col3, col4 = st.columns([0.6,0.6,0.8,1])
                        with col1:
                            if len(st.session_state["Variables"]) != 0:
                                list_combination_features = st.selectbox('Select the variable', list(st.session_state["Variables"]), key='list_combination_features_cluster')
                                combinations_features = st.session_state[list_combination_features]
                            else:
                                st.warning("There is not feature combination variable")
                        with col2:
                            if len(st.session_state["Variables"]) != 0:
                                remove_vars_combinations = st.selectbox("Remove variables of combinations?", ['No','Yes'], key='remove_vars_combinations_cluster')
                        with col3:
                            if remove_vars_combinations != 'No':
                                list_remove_vars_combinations = st.multiselect("Select the variables to remove of combinations", st.session_state[var_dataset_cluster].columns, key='list_remove_vars_combinations_cluster')
                    elif valid_features_cluster == 'Remove variables':
                        col1, col2 = st.columns([1.35,0.66])
                        with col1:
                            remov_var_cluster = st.multiselect("Select the variables to remove", st.session_state[var_dataset_cluster].columns, key='remov_var_cluster')

                    col1, col2, col3, col4 = st.columns([0.5,0.7,0.8,1])
                    with col1:
                        select_type_k_cluster = st.selectbox("Select a list or range of K", ['Range','List'], key='select_type_k_cluster')
                    with col2:
                        if select_type_k_cluster == 'Range':
                            number_of_k_cluster = st.number_input("Input the k value", min_value=3, step=1, value=3, key='number_of_k_cluster')
                        else:
                            number_of_k_cluster = st.text_input("Input values separated by comma, ex: 3,4,5", key='number_of_k_cluster')
                            if number_of_k_cluster != '':
                                number_of_k_cluster = [int(num) for num in number_of_k_cluster.split(',')]
                    with col3:
                        name_experiment_cluster = st.text_input("Input name of experiment")

                    button_cluster = st.button("Apply", key='button_cluster')
                    st.write("--------------------------")
                    if button_cluster:

                        from sklearn.preprocessing import StandardScaler

                        if valid_features_cluster == 'None':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster])
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster].columns)
                            else:
                                X =  st.session_state[var_dataset_cluster]
                        elif valid_features_cluster == 'Select variables':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster][select_features_cluster])
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster][select_features_cluster].columns)
                            else:
                                X = st.session_state[var_dataset_cluster][select_features_cluster]
                        #elif valid_features_cluster == 'Combination features':
                        #    if go_scaler == 'Yes':
                        #        if remove_vars_combinations == 'Yes':
                        #            X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster].drop(list_remove_vars_combinations,axis=1))
                        #            X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster].drop(list_remove_vars_combinations,axis=1).columns)
                        #        else:
                        #            X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster])
                        #            X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster].columns)
                        #    else:
                        #        if remove_vars_combinations == 'Yes':
                        #            X = st.session_state[var_dataset_cluster].drop(list_remove_vars_combinations,axis=1)
                        #        else:
                        #            X = st.session_state[var_dataset_cluster]
                        elif valid_features_cluster == 'Remove variables':
                            if go_scaler == 'Yes':
                                X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster].drop(remov_var_cluster,axis=1))
                                X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster].drop(remov_var_cluster,axis=1).columns)
                            else:
                                X = st.session_state[var_dataset_cluster].drop(remov_var_cluster,axis=1)

                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_samples, silhouette_score
                        import uuid

                        if combinations_features:
                            new_experiment_cluster = {}
                            c = 0
                            for combination in combinations_features:

                                if valid_features_cluster == 'Combination features':
                                    if go_scaler == 'Yes':
                                        if remove_vars_combinations == 'Yes':
                                            X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster][combination].drop(list_remove_vars_combinations,axis=1))
                                            X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster][combination].drop(list_remove_vars_combinations,axis=1).columns)
                                        else:
                                            X = StandardScaler().fit_transform(st.session_state[var_dataset_cluster][combination])
                                            X = pd.DataFrame(X, columns=st.session_state[var_dataset_cluster][combination].columns)
                                    else:
                                        if remove_vars_combinations == 'Yes':
                                            X = st.session_state[var_dataset_cluster][combination].drop(list_remove_vars_combinations,axis=1)
                                        else:
                                            X = st.session_state[var_dataset_cluster][combination]

                                st.subheader(f"Combination {combination}")

                                for i,k in enumerate(range(2, number_of_k_cluster+1) if select_type_k_cluster == 'Range' else number_of_k_cluster):
                                    model = KMeans(k, random_state=42)
                                    cluster_labels = model.fit_predict(X[combination])

                                    #var_silhouette_score = silhouette_score(X[combination], cluster_labels)
                                    #st.write(f"**The mean silhouette score for k = {k} is {round(var_silhouette_score,4)}**")
                                    st.write(f"**The inertia score for k = {k} is {round(model.inertia_,4)}**")

                                    new_experiment_cluster[c+i] = {}
                                    new_experiment_cluster[c+i]['name_experiment'] = name_experiment_cluster
                                    new_experiment_cluster[c+i]['id_experiment'] = str(uuid.uuid4())
                                    new_experiment_cluster[c+i]['columns'] = list(X[combination].columns)
                                    new_experiment_cluster[c+i]['df_shape'] = [X[combination].shape[0], X[combination].shape[1]]
                                    new_experiment_cluster[c+i]['n_clusters'] = model.n_clusters
                                    new_experiment_cluster[c+i]['centroids'] = model.cluster_centers_.tolist()
                                    new_experiment_cluster[c+i]['inertia_score'] = model.inertia_
                                    #new_experiment_cluster[c+i]['silhouette_score'] = var_silhouette_score
                                    new_experiment_cluster[c+i]['silhouette_score'] = 0 
                                    new_experiment_cluster[c+i]['labels'] = model.labels_

                                    st.write("--------------------------")

                                c = len(new_experiment_cluster)

                            #id = str(uuid.uuid4()).replace('-','')
                            with open(f'unsupervised_learning/Kmeans_experiments/{name_experiment_cluster}.pkl', 'wb') as f:
                                pickle.dump(new_experiment_cluster, f)

                        else:
                            
                            #for _, _, files in os.walk('unsupervised_learning/Kmeans_experiments/'):
                            #    Kmeans_experiments = files
                            #list_Kmeans_experiments.insert(0,'None')
                            #teste = st.selectbox("files", list_Kmeans_experiments)
                            #st.write(Kmeans_experiments)
                            #list_Kmeans_experiments = [int(name.replace('.pkl','')) for name in Kmeans_experiments]
                            #st.write(list_Kmeans_experiments)

                            st.subheader(f"Features {list(X.columns)}")

                            new_experiment_cluster = {}
                            for i,k in enumerate(range(2, number_of_k_cluster+1) if select_type_k_cluster == 'Range' else number_of_k_cluster):
                                model = KMeans(k, random_state=42)
                                cluster_labels = model.fit_predict(X)

                                #var_silhouette_score = silhouette_score(X, cluster_labels)
                                #st.write(f"**The mean silhouette score for k = {k} is {round(var_silhouette_score,4)}**")
                                st.write(f"**The inertia score for k = {k} is {round(model.inertia_,4)}**")

                                new_experiment_cluster[0+i] = {}
                                new_experiment_cluster[0+i]['name_experiment'] = name_experiment_cluster
                                new_experiment_cluster[0+i]['id_experiment'] = str(uuid.uuid4())
                                new_experiment_cluster[0+i]['columns'] = list(X.columns)
                                new_experiment_cluster[0+i]['df_shape'] = [X.shape[0], X.shape[1]]
                                new_experiment_cluster[0+i]['n_clusters'] = model.n_clusters
                                new_experiment_cluster[0+i]['centroids'] = model.cluster_centers_.tolist()
                                new_experiment_cluster[0+i]['inertia_score'] = model.inertia_
                                #new_experiment_cluster[0+i]['silhouette_score'] = var_silhouette_score
                                new_experiment_cluster[0+i]['silhouette_score'] = 0
                                new_experiment_cluster[0+i]['labels'] = model.labels_

                                st.write("-----------------------")

                            #id = str(uuid.uuid4()).replace('-','')
                            with open(f'unsupervised_learning/Kmeans_experiments/{name_experiment_cluster}.pkl', 'wb') as f:
                                pickle.dump(new_experiment_cluster, f)

            elif TypeClusteringModel == 'DBSCAN':
                st.warning("Coming soon!")

            elif TypeClusteringModel == 'MeanShift':
                st.warning("Coming soon!")

            elif TypeClusteringModel == 'AgglomerativeClustering':
                st.warning("Coming soon!")

    def appReportExperiments():

        st.title('Report experiments')

        TypeMachineLearningReportExperiments = option_menu('', ['Supervised Learning','Unsupervised Learning'], 
            default_index=0, orientation="horizontal",
            styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                    "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
        )

        if TypeMachineLearningReportExperiments == 'Supervised Learning':

            experiments = mlflow.list_experiments()
            list_name_of_experiments = [experiments.name for experiments in experiments]
            list_name_of_experiments.insert(0, 'None')

            optionExperimentsName = st.selectbox(
            'Select the experiment',
            (list_name_of_experiments))

            if experiments:
                if optionExperimentsName != 'None':

                    NameExperiment = optionExperimentsName

                    #@st.cache
                    def infos_experiment(nameExperiment):
                        experiment_id_info = mlflow.get_experiment_by_name(nameExperiment).experiment_id
                        list_ids_runs_info =  [x.run_id for x in mlflow.list_run_infos(experiment_id_info) if x.status == 'FINISHED']
                        completed_operations_info = len([x for x in mlflow.list_run_infos(experiment_id_info) if x.status == 'FINISHED'])
                        failed_operations_info = len([x for x in mlflow.list_run_infos(experiment_id_info) if x.status == 'FAILED'])
                        return experiment_id_info,list_ids_runs_info,completed_operations_info,failed_operations_info

                    experiment_id,list_ids_runs,completed_operations,failed_operations = infos_experiment(NameExperiment)

                    if list_ids_runs:

                        st.write(
                        f"""

                        Infos:
                        ---------------
                        - **Experiment name**: {NameExperiment}
                        - **Experiment id**: {experiment_id}
                        - **Total operations**: {completed_operations + failed_operations}
                        - **Completed operations**: {completed_operations}
                        - **Failed operations**: {failed_operations}
                        """
                        )

                        @st.cache
                        def func_df_runs(runs):
                            df = pd.DataFrame()
                            for id_run in list_ids_runs:
                                run = mlflow.get_run(id_run)
                                #with open(f'mlruns/{experiment_id}/{id_run}/artifacts/columns.txt') as f:
                                #    json_data = json.load(f)
                                
                                dic_metrics = run.data.metrics
                                dic_metrics['time_end'] = round((run.info.end_time - run.info.start_time) / 1000,2)
                                dic_metrics['Algorithm'] = run.data.tags['mlflow.runName']
                                dic_metrics['id_run'] = id_run
                                dic_metrics['Columns'] = run.data.params['Features']
                                dic_metrics['Target'] = run.data.params['Target']
                                #dic_metrics['Columns'] = json_data['columns']
                                #dic_metrics['Target'] = json_data['target']
                                
                                infos_runs = pd.DataFrame([dic_metrics])
                                df = pd.concat([df, infos_runs], axis=0)
                            df.reset_index(drop=True, inplace=True)
                            return df
                    
                        df_runs = func_df_runs(list_ids_runs)

                        with st.expander("List of runs", expanded=False):

                            show_list_runs = st.checkbox("Show me", key='show_list_runs')
                            if show_list_runs:

                                #list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']

                                #df_runs = pd.DataFrame()
                                #for id_run in list_ids_runs:
                                #    run = mlflow.get_run(id_run)
                                #    #with open(f'mlruns/{experiment_id}/{id_run}/artifacts/columns.txt') as f:
                                #    #    json_data = json.load(f)
                                #    
                                #    dic_metrics = run.data.metrics
                                #    dic_metrics['time_end'] = round((run.info.end_time - run.info.start_time) / 1000,2)
                                #    dic_metrics['Algorithm'] = run.data.tags['mlflow.runName']
                                #    dic_metrics['id_run'] = id_run
                                #    dic_metrics['Columns'] = run.data.params['Features']
                                #    dic_metrics['Target'] = run.data.params['Target']
                                #    #dic_metrics['Columns'] = json_data['columns']
                                #    #dic_metrics['Target'] = json_data['target']
                                #    
                                #    infos_runs = pd.DataFrame([dic_metrics])
                                #    df_runs = pd.concat([df_runs, infos_runs], axis=0)
                                #df_runs.reset_index(drop=True, inplace=True)

                                cols_to_filter = df_runs.drop(['X train','X test','Algorithm','id_run','Columns','Target'],axis=1).columns
                                col1, col2 = st.columns([0.5,2])
                                with col1:
                                    order = st.selectbox("Ascending", [False,True])
                                with col2:
                                    orderbyDF = st.multiselect("Order By", cols_to_filter)

                                # -----------------------------------------------------------------------------------------------
                                cols_to_filter_2 = df_runs.drop(['Algorithm','Columns','Target'],axis=1).columns
                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    filterbyDF_variable = st.selectbox("Select the variable", cols_to_filter_2)
                                with col2:
                                    filterbyDF_controller = st.selectbox("", ["<", "=", ">", "=="])
                                with col3:
                                    if filterbyDF_controller == '==':
                                        filterbyDF_value_compare = st.text_input("Input")
                                    else:
                                        filterbyDF_value = st.number_input("Value", min_value=0, value=0, step=1)

                                col1, col2, col3 = st.columns([0.5,0.5,3.8])
                                with col1:
                                    submit_filter = st.button("Submit filter")
                                with col2:
                                    submit_filter_reset = st.button("Reset filter")
                                #------------------------------------------------------------------------------------------------

                                df_runs_filter = df_runs.sort_values(by=orderbyDF, ascending=order)

                                if submit_filter:
                                    if filterbyDF_controller == '<':
                                        df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] < filterbyDF_value]
                                    elif filterbyDF_controller == '=':
                                        df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] == filterbyDF_value]
                                    elif filterbyDF_controller == '>':
                                        df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] > filterbyDF_value]

                                if submit_filter_reset:
                                    st.experimental_rerun()

                                #cm = sns.light_palette("green", as_cmap=True)
                                cm = sns.color_palette("Blues", as_cmap=True)
                                #cm = sns.dark_palette("#69d", reverse=False, as_cmap=True)
                                st.dataframe(df_runs_filter[['Algorithm','Accuracy','AUC','F1','Kappa','Precision','Recall','Specificity','TN','TP','FN','FP','Nº columns','X test','X train','time_end','Columns','Target','id_run']].style.background_gradient(cmap=cm))
                                #st.write(df_runs_filter.columns)

                                st.write("-------------------------------------")
                                filter_to_barplot = st.selectbox("Select the variable", cols_to_filter.insert(0, 'None'))
                                if filter_to_barplot != 'None':
                                    
                                    fig = px.bar(df_runs.sort_values(by=filter_to_barplot, ascending=True), x=filter_to_barplot, y="id_run")
                                    fig.update_yaxes(visible=True, showticklabels=False)
                                    st.plotly_chart(fig)
                                else:
                                    st.write("Select the variable to plotting")
                        
                        with st.expander("Charts", expanded=False):

                            eda_plot = EDA.EDA(df_runs)

                            def plot_multivariate(obj_plot, radio_plot):
                                
                                if 'Boxplot' in radio_plot:
                                    st.subheader('Boxplot')
                                    col_y  = st.multiselect("Choose main variable (numerical)",obj_plot.num_vars, key ='boxplot_multivariate')
                                    #col_x  = st.selectbox("Choose x variable (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                                    hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                                    #if st.sidebar.button('Plot boxplot chart'):
                                    show_boxplot_chart_report = st.checkbox("Show me", key='show_boxplot_chart_report')
                                    if show_boxplot_chart_report:
                                        st.plotly_chart(obj_plot.box_plot(col_y, hue_opt))
                                
                                if 'Histogram' in radio_plot:
                                    st.subheader('Histogram')
                                    col_hist = st.selectbox("Choose main variable", obj_plot.num_vars, key = 'hist')
                                    hue_opt = st.selectbox("Hue (categorical) *optional",obj_plot.columns.insert(0,None), key = 'hist')
                                    show_histogram_chart_report = st.checkbox("Show me", key='show_histogram_chart_report')
                                    if show_histogram_chart_report:
                                        bins_, range_ = None, None
                                        bins_ = st.slider('Number of bins *optional', value = 30)
                                        range_ = st.slider('Choose range *optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                                                (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
                                        #if st.button('Plot histogram chart'):
                                        st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

                                if 'Scatterplot' in radio_plot: 
                                    st.subheader('Scatter plot')
                                    col_x = st.selectbox("Choose X variable (numerical)", obj_plot.num_vars, key = 'scatter')
                                    col_y = st.selectbox("Choose Y variable (numerical)", obj_plot.num_vars, key = 'scatter')
                                    hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None), key = 'scatter')
                                    size_opt = st.selectbox("Size (numerical) *optional",obj_plot.columns.insert(0,None), key = 'scatter')
                                    hover_data = st.multiselect("Hover data *optional",obj_plot.columns, key = 'scatter')
                                    #if st.sidebar.button('Plot scatter chart'):
                                    show_scatterplot_chart_report = st.checkbox("Show me", key='show_scatterplot_chart_report')
                                    if show_scatterplot_chart_report:
                                        st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt, hover_data))

                            radio_plot = st.multiselect('Choose plot style', ['Boxplot', 'Histogram', 'Scatterplot'])

                            plot_multivariate(eda_plot, radio_plot)

                        with st.expander("Machine learning evaluation metrics", expanded=False):
                            
                            col1, col2, col3 = st.columns([1,1,1])
                            with col1:
                                compare_or_view = st.selectbox("Compare metrics between models or see all metrics in a model",['None','View metrics in model','Compare model metrics'])

                            if compare_or_view == 'View metrics in model':

                                with st.form(key='my_form_view_metrics_model', clear_on_submit=True):

                                    id_model = st.selectbox("Select the model", list_ids_runs)

                                    #button_show_view_metrics = st.button("View", key='button_show_view_metrics')
                                    button_show_view_metrics = st.form_submit_button('Submit')

                                    if button_show_view_metrics:
                                        with st.spinner('Wait for it...'):
                                            import scikitplot as skplt
                                            info_run = mlflow.get_run(id_model)

                                            used_model_name = info_run.data.tags['mlflow.runName']
                                            #st.write("Name Model", used_model_name)
                                            used_columns = info_run.data.params['Features'].split(',')
                                            #st.write("Used columns", used_columns)
                                            used_target = info_run.data.params['Target']
                                            #st.write("Name of target", used_target)

                                            y_test = [int(x) for x in info_run.data.params["y_test"].split(',')]
                                            #st.write("y_test",y_test)
                                            y_pred_prob_class_1 = [float(x) for x in info_run.data.params["y_pred_prob_class_1"].split(',')]
                                            #st.write("y_pred_prob_class_1",y_pred_prob_class_1)
                                            y_pred_prob_class_0 = [float(x) for x in info_run.data.params["y_pred_prob_class_0"].split(',')]
                                            #st.write("y_pred_prob_class_0",y_pred_prob_class_0)

                                            y_pred = []
                                            for class_0,class_1 in zip(y_pred_prob_class_0,y_pred_prob_class_1):
                                                y_pred.append([class_0,class_1])
                                            y_pred = np.array(y_pred)
                                            #st.write("y_pred",y_pred)

                                            y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_prob_class_1]
                                            #st.write("y_pred_binary",y_pred_binary)

                                            #dic_metrics = run.data.metrics
                                            #dic_metrics['time_end'] = round((run.info.end_time - run.info.start_time) / 1000,2)
                                            #dic_metrics['Algorithm'] = run.data.tags['mlflow.runName']
                                            #dic_metrics['id_run'] = id_run
                                            #dic_metrics['Columns'] = run.data.params['Features']
                                            #dic_metrics['Target'] = run.data.params['Target']
                                            metrics = pd.DataFrame(columns=['Metric','Value'])
                                            metrics.loc[0,'Metric'] = 'Accuracy'
                                            metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                            metrics.loc[1,'Metric'] = 'AUC'
                                            metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                            metrics.loc[2,'Metric'] = 'F1'
                                            metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                            metrics.loc[3,'Metric'] = 'Kappa'
                                            metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                            metrics.loc[4,'Metric'] = 'Precision'
                                            metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                            metrics.loc[5,'Metric'] = 'Recall'
                                            metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                            metrics.loc[6,'Metric'] = 'Specificity'
                                            metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']

                                            st.write(
                                            f"""
                                            - **Dataset rows:** {info_run.data.metrics['X train'] + info_run.data.metrics['X test']}
                                            - **Dataset columns:** {len(used_columns)}
                                            - **X train rows:** {info_run.data.metrics['X train']}
                                            - **X test rows:** {info_run.data.metrics['X test']}
                                            - **Used columns:** {used_columns}
                                            - **Target name:** {used_target}
                                            - **Model:** {df_runs[df_runs['id_run']==id_model]['Algorithm'].values[0]} 
                                            """
                                            )

                                            st.write("")

                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                st.table(metrics)
                                            with col2:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                                st.pyplot(fig)
                                            with col3:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                                st.pyplot(fig)

                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            with col2:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            with col3:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)

                                            col1, col2, col3 = st.columns([1,1,1])
                                            with col1:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            with col2:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            with col3:
                                                probas_list = [y_pred]
                                                clf_names = [used_model_name]
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                                st.pyplot(fig)

                                            df_prob = pd.DataFrame(y_test, columns=[used_target])
                                            df_prob['y_pred'] = y_pred_binary
                                            df_prob['prob'] = y_pred_prob_class_1
                                            breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                            labels = list(range(1,21))
                                            df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                            
                                            analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                                .agg({used_target:'count'})\
                                                                                                .rename(columns={used_target:'class_0'})

                                            analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                                .agg({used_target:'count'})\
                                                                                                .rename(columns={used_target:'class_1'})
                                            analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                            analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                            analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                            analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                            st.table(analysis_prob_full)

                            elif compare_or_view == 'Compare model metrics':

                                col1, col2 = st.columns([1,1])
                                with col1:
                                    id_model = st.selectbox("Select the id model on the left", list_ids_runs, key='id_model_compare_model_metrics_1')
                                    type_compare = st.multiselect("Select metric type on the left",['Metric table',
                                            'Confusion matrix without normalization','Confusion matrix with normalization',
                                            'ROC plot','KS plot','Precision X Recall plot','Cumulative plot','Lift curve plot',
                                            'Calibration curve plot','Range prob table'], key='type_compare_compare_model_metrics_1')

                                    if len(type_compare) > 0:
                                        with st.spinner('Wait for it...'):
                                            import scikitplot as skplt
                                            info_run = mlflow.get_run(id_model)

                                            y_test = [int(x) for x in info_run.data.params["y_test"].split(',')]
                                            y_pred_prob_class_1 = [float(x) for x in info_run.data.params["y_pred_prob_class_1"].split(',')]
                                            y_pred_prob_class_0 = [float(x) for x in info_run.data.params["y_pred_prob_class_0"].split(',')]

                                            y_pred = []
                                            for class_0,class_1 in zip(y_pred_prob_class_0,y_pred_prob_class_1):
                                                y_pred.append([class_0,class_1])
                                            y_pred = np.array(y_pred)

                                            y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_prob_class_1]

                                            if 'Metric table' in type_compare:
                                                metrics = pd.DataFrame(columns=['Metric','Value'])
                                                metrics.loc[0,'Metric'] = 'Accuracy'
                                                metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                                metrics.loc[1,'Metric'] = 'AUC'
                                                metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                                metrics.loc[2,'Metric'] = 'F1'
                                                metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                                metrics.loc[3,'Metric'] = 'Kappa'
                                                metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                                metrics.loc[4,'Metric'] = 'Precision'
                                                metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                                metrics.loc[5,'Metric'] = 'Recall'
                                                metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                                metrics.loc[6,'Metric'] = 'Specificity'
                                                metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']
                                                st.table(metrics)
                                            if 'Confusion matrix without normalization' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                                st.pyplot(fig)
                                            if 'Confusion matrix with normalization' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                                st.pyplot(fig)
                                            if 'ROC plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'KS plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Precision X Recall plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Cumulative plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Lift curve plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Calibration curve plot' in type_compare:
                                                probas_list = [y_pred]
                                                clf_names = [used_model_name]
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                                st.pyplot(fig)
                                            if 'Range prob table' in type_compare:
                                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                                df_prob['y_pred'] = y_pred_binary
                                                df_prob['prob'] = y_pred_prob_class_1
                                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                                labels = list(range(1,21))
                                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                                
                                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                                    .agg({used_target:'count'})\
                                                                                                    .rename(columns={used_target:'class_0'})

                                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                                    .agg({used_target:'count'})\
                                                                                                    .rename(columns={used_target:'class_1'})
                                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                                st.table(analysis_prob_full)

                                with col2:
                                    id_model = st.selectbox("Select the id model on the right", list_ids_runs, key='id_model_compare_model_metrics_2')
                                    type_compare = st.multiselect("Select metric type on the right",['Metric table',
                                            'Confusion matrix without normalization','Confusion matrix with normalization',
                                            'ROC plot','KS plot','Precision X Recall plot','Cumulative plot','Lift curve plot',
                                            'Calibration curve plot','Range prob table'], key='type_compare_compare_model_metrics_2')

                                    if len(type_compare) > 0:
                                        with st.spinner('Wait for it...'):
                                            import scikitplot as skplt
                                            info_run = mlflow.get_run(id_model)

                                            y_test = [int(x) for x in info_run.data.params["y_test"].split(',')]
                                            y_pred_prob_class_1 = [float(x) for x in info_run.data.params["y_pred_prob_class_1"].split(',')]
                                            y_pred_prob_class_0 = [float(x) for x in info_run.data.params["y_pred_prob_class_0"].split(',')]

                                            y_pred = []
                                            for class_0,class_1 in zip(y_pred_prob_class_0,y_pred_prob_class_1):
                                                y_pred.append([class_0,class_1])
                                            y_pred = np.array(y_pred)

                                            y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred_prob_class_1]

                                            
                                            if 'Metric table' in type_compare:
                                                metrics = pd.DataFrame(columns=['Metric','Value'])
                                                metrics.loc[0,'Metric'] = 'Accuracy'
                                                metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                                metrics.loc[1,'Metric'] = 'AUC'
                                                metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                                metrics.loc[2,'Metric'] = 'F1'
                                                metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                                metrics.loc[3,'Metric'] = 'Kappa'
                                                metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                                metrics.loc[4,'Metric'] = 'Precision'
                                                metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                                metrics.loc[5,'Metric'] = 'Recall'
                                                metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                                metrics.loc[6,'Metric'] = 'Specificity'
                                                metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']
                                                st.table(metrics)
                                            if 'Confusion matrix without normalization' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                                st.pyplot(fig)
                                            if 'Confusion matrix with normalization' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                                st.pyplot(fig)
                                            if 'ROC plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'KS plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Precision X Recall plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Cumulative plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Lift curve plot' in type_compare:
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                                st.pyplot(fig)
                                            if 'Calibration curve plot' in type_compare:
                                                probas_list = [y_pred]
                                                clf_names = [used_model_name]
                                                fig, ax = plt.subplots()
                                                skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                                st.pyplot(fig)
                                            if 'Range prob table' in type_compare:
                                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                                df_prob['y_pred'] = y_pred_binary
                                                df_prob['prob'] = y_pred_prob_class_1
                                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                                labels = list(range(1,21))
                                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                                
                                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                                    .agg({used_target:'count'})\
                                                                                                    .rename(columns={used_target:'class_0'})

                                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                                    .agg({used_target:'count'})\
                                                                                                    .rename(columns={used_target:'class_1'})
                                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                                st.table(analysis_prob_full)

                    else:
                        st.warning("Don't have experiments")

                else:
                    st.write("Select the experiment!")
            else:
                st.write("There is no experiments loaded")

        elif TypeMachineLearningReportExperiments == 'Unsupervised Learning':
            
            for _, _, files in os.walk('unsupervised_learning/Kmeans_experiments/'):
                list_files = files

            list_name_of_experiments_unsupervised = list_files.copy()

            optionExperimentsNameUnsupervised = st.multiselect(
            'Select the experiment',
            (list_name_of_experiments_unsupervised))

            if optionExperimentsNameUnsupervised:

                #if optionExperimentsNameUnsupervised == 'All':
                df_infos_models_unsupervised = pd.DataFrame()
                for file in optionExperimentsNameUnsupervised:
                    with open(f'unsupervised_learning/Kmeans_experiments/{file}', 'rb') as f:
                        loaded_dict = pickle.load(f)
                    
                    for i in range(0,len(loaded_dict)):
                        df_temp = pd.DataFrame([loaded_dict[i]])
                        df_infos_models_unsupervised = pd.concat([df_infos_models_unsupervised,df_temp],axis=0)
                df_infos_models_unsupervised.reset_index(inplace=True, drop=True)

                st.write(
                f"""
                Infos:
                ---------------
                - **Total of experiments**: {len(df_infos_models_unsupervised['name_experiment'].value_counts())}
                - **Total operations**: {df_infos_models_unsupervised.shape[0]}
                """
                )

                with st.expander("List of runs", expanded=False):

                    show_df_clusters = st.checkbox("Show me", key='show_df_clusters')
                    if show_df_clusters:
                        st.dataframe(df_infos_models_unsupervised)

                with st.expander("Clusters analysis", expanded=False):

                    show_cluster_analysis = st.checkbox("Show me", key='show_cluster_analysis')
                    if show_cluster_analysis:

                        col1, col2, col3, col4 = st.columns([1,1,1,1])
                        with col1:
                            metric_analysis_filter = st.selectbox("Filter evaluation metric",['WCSS','Silhouette'])
                        with col2:
                            name_experiments_filter = st.multiselect("Filter the experiments", df_infos_models_unsupervised['name_experiment'].unique().tolist())
                        with col3:
                            number_cluster_filter = st.multiselect("Filter the number of clusters", df_infos_models_unsupervised['n_clusters'].unique().tolist())
                        with col4:
                            number_columns_filter = st.multiselect("Filter the number of columns", df_infos_models_unsupervised['columns'].apply(lambda x: len(x)).unique().tolist())
                        
                        col1, col2, col3, col4 = st.columns([1,1,1,1])
                        with col1:
                            var_height = st.number_input("Graph height", min_value=0, value=200*len(df_infos_models_unsupervised['n_clusters'].unique().tolist()))

                        if metric_analysis_filter == 'WCSS':
                            metric = 'inertia_score'
                        elif metric_analysis_filter == 'Silhouette':
                            metric = 'silhouette_score'

                        df = df_infos_models_unsupervised.copy()
                        if name_experiments_filter:
                            df = df[df['name_experiment'].isin(name_experiments_filter)]
                        if number_cluster_filter:
                            df = df[df['n_clusters'].isin(number_cluster_filter)]
                        if number_columns_filter:
                            df['number_columns'] = df['columns'].apply(lambda x: len(x))
                            df = df[df['number_columns'].isin(number_columns_filter)]
     
                        df.sort_values(by='n_clusters', inplace=True)
                            
                        total_rows_subplots = len(df['n_clusters'].unique().tolist())
                        min_range_plot = 0
                        max_range_plot = df[metric].max() * 1.50

                        fig = make_subplots(rows=total_rows_subplots, cols=1, shared_xaxes=False, vertical_spacing=0.005)
                        for i,n in enumerate(df['n_clusters'].unique().tolist()):
                            fig.append_trace(
                                go.Scatter( x= df[df['n_clusters']==n]['id_experiment'],
                                            y= df[df['n_clusters']==n][metric],
                                            name = f"Cluster {n}"),

                                    row=1+i, col=1)
                            fig.update_yaxes(title_text=f"{n} cluster", range=[min_range_plot, max_range_plot], row=1+i, col=1, showgrid=False)
                            
                        fig.update_xaxes(visible=False)
                        fig.update_layout(height=var_height, showlegend=False,hovermode="x unified",
                                        title_text="Cluster Analysis")
                        st.plotly_chart(fig, use_container_width=True)

                #with st.expander("teste", expanded=False):

                    #df_DD = df_infos_models_unsupervised.copy()
                    #st.dataframe(df_infos_models_unsupervised.drop(['labels'],axis=1))
                    #st.write(df_infos_models_unsupervised['columns'][0])
                    #st.write(df_infos_models_unsupervised['columns'][5])

                with st.expander("Data description", expanded=False):

                    col1, col2, col3, col4 = st.columns([1,1,1,1])
                    with col1:
                        dataset_data_description = st.selectbox("Select the dataset", st.session_state['dataset'])
                    with col2:
                        name_experiments_filter_DD = st.multiselect("Filter the experiments", df_infos_models_unsupervised['name_experiment'].unique().tolist(), key='name_experiments_filter_DD')
                    with col3:
                        number_cluster_filter_DD = st.multiselect("Filter the number of clusters", df_infos_models_unsupervised['n_clusters'].unique().tolist(), key='number_cluster_filter_DD')
                    with col4:
                        number_columns_filter_DD = st.multiselect("Filter the number of columns",df_infos_models_unsupervised['columns'].apply(lambda x: len(x)).unique().tolist(), key='number_columns_filter_DD')

                    #button_DD = st.button("Apply", key='button_DD')
                    show_mm = st.checkbox("show me describe", key='show_mm')
                    if show_mm:

                        df_dataset = st.session_state[dataset_data_description].copy()
                        cols_origem = df_dataset.columns.tolist()

                        df_DD = df_infos_models_unsupervised.copy()
                        if name_experiments_filter_DD:
                            df_DD = df_DD[df_DD['name_experiment'].isin(name_experiments_filter_DD)]
                            df_DD.reset_index(inplace=True, drop=True)
                        if number_cluster_filter_DD:
                            df_DD = df_DD[df_DD['n_clusters'].isin(number_cluster_filter_DD)]
                            df_DD.reset_index(inplace=True, drop=True)
                        if number_columns_filter_DD:
                            df_DD['number_columns'] = df_DD['columns'].apply(lambda x: len(x))
                            df_DD = df_DD[df_DD['number_columns'].isin(number_columns_filter_DD)]
                            df_DD.reset_index(inplace=True, drop=True)

                        for i in range(0,len(df_DD)):
                            df_dataset['DCluster__'+ df_DD.iloc[i]['id_experiment']] = df_DD.iloc[i]['labels'].tolist()
                        
                        cols_groupby = []
                        for col in df_dataset.columns:
                            if col[0:10] == 'DCluster__':
                                cols_groupby.append(col)

                        #st.dataframe(df_DD)
                        #st.dataframe(df_dataset)

                        df_describe_all = pd.DataFrame()
                        for i in range(0,len(df_DD)):
                            list_cols_useded = ', '.join(df_DD['columns'][i])
                            qnt_cluster = len(df_dataset[cols_groupby[i]].value_counts())

                            temp_describe = None
                            temp_describe = df_dataset.groupby(cols_groupby[i], as_index=False)[cols_origem].describe().T
                            temp_describe.reset_index(inplace=True)
                            temp_describe.columns = ['Cluster '+str(col) if type(col) == int else 'Column' if col=='level_0' else 'Descriptive' for col in temp_describe.columns]
                            
                            #temp_describe_count = iris.groupby(cols_groupby[i], as_index=False)[['Sepal.Length','Sepal.Width']].describe().T
                            #temp_describe_count.reset_index(inplace=True)
                            #temp_describe_count.columns = ['Cluster '+str(col) if type(col) == int else 'Column' if col=='level_0' else 'Descriptive' for col in temp_describe_count.columns]
                            describe_count = temp_describe[temp_describe['Descriptive']=='count']
                            describe_count = describe_count[describe_count.index==0]
                            describe_count = pd.concat([describe_count,describe_count],axis=0)
                            describe_count.reset_index(inplace=True, drop=True)

                            for col in describe_count.columns:
                                for i in range(0,len(describe_count)):
                                    if col == 'Column':
                                        if i == 0:
                                            describe_count.loc[i,col] = 'Count(%)'
                                        else:
                                            describe_count.loc[i,col] = 'Count'
                                    elif col == 'Descriptive':
                                        if i == 0:
                                            describe_count.loc[i,col] = 'count(%)'
                                        else:
                                            describe_count.loc[i,col] = 'count'
                                    else:
                                        if i == 0:
                                            describe_count.loc[i,col] = round((describe_count[col][i] / df_dataset.shape[0])*100,2)
                                        else:
                                            describe_count.loc[i,col] = describe_count[col][i]
                            
                            #print(describe_count)
                            #temp_describe_all = iris.groupby(cols_groupby[i], as_index=False)[['Sepal.Length','Sepal.Width']].describe().T
                            #temp_describe_all.reset_index(inplace=True)
                            #temp_describe_all.columns = ['Cluster '+str(col) if type(col) == int else 'Column' if col=='level_0' else 'Descriptive' for col in temp_describe_all.columns]
                            
                            #temp_describe_all = temp_describe[temp_describe['Descriptive']!='count']
                            data_temp_all = pd.concat([describe_count, temp_describe[temp_describe['Descriptive']!='count']],axis=0)
                            data_temp_all['number_clusters'] = qnt_cluster
                            data_temp_all['columns_in_clusters'] = list_cols_useded

                            data_temp_all.reset_index(inplace=True, drop=True)

                            df_describe_all = pd.concat([df_describe_all, data_temp_all],axis=0)
                            df_describe_all.reset_index(inplace=True, drop=True)
                            
                            #teste = list_cols_useded.split(', ')
                            #st.write(teste)
                            #st.write(list_cols_useded)
                        #for i in range(0,len(df_DD)):
                        #    st.write(df_DD['columns'][i])
                        order_by_cols = df_describe_all.columns.tolist()
                        order_by_cols.remove('columns_in_clusters')
                        order_by_cols.append('columns_in_clusters')
                        order_by_cols.remove('number_clusters')
                        order_by_cols.append('number_clusters')
                        df_describe_all = df_describe_all[order_by_cols]

                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:
                            filter_clusters_df_describe = st.multiselect("Filter clusters", df_describe_all['number_clusters'].unique().tolist(), default=df_describe_all['number_clusters'].unique().tolist())
                        with col2:
                            filter_columns_df_describe = st.multiselect("Filter columns", df_describe_all['Column'].unique().tolist(), default=df_describe_all['Column'].unique().tolist())
                        with col3:
                            filter_agg_df_describe = st.multiselect("Filter summary", df_describe_all['Descriptive'].unique().tolist(), default=df_describe_all['Descriptive'].unique().tolist())
                        
                        filter_col_view = st.multiselect("Filter view", df_describe_all.drop(['number_clusters'],axis=1).columns.tolist(), default=df_describe_all.drop(['number_clusters'],axis=1).columns.tolist())

                        df_describe_all = df_describe_all[df_describe_all['number_clusters'].isin(filter_clusters_df_describe)]
                        df_describe_all = df_describe_all[df_describe_all['Column'].isin(filter_columns_df_describe)]
                        df_describe_all = df_describe_all[df_describe_all['Descriptive'].isin(filter_agg_df_describe)]

                        subset_cols_style = df_describe_all.columns.tolist()
                        subset_cols_style.remove('Column')
                        subset_cols_style.remove('Descriptive')
                        subset_cols_style.remove('number_clusters')
                        
                        #st.dataframe(df_describe_all[filter_col_view].style.background_gradient(cmap=sns.color_palette("Blues", as_cmap=True), axis=1))
                        st.dataframe(df_describe_all[filter_col_view])

                        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                        with col1:
                            #st.write("Downdload")
                            def convert_df(df):
                                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                                return df.to_csv(index=False, sep=';')

                            st.download_button(
                                label="Download data as CSV",
                                data=convert_df(df_describe_all),
                                file_name="describe_clustering.csv",
                                mime='text/csv',
                            )
            else:
                    st.write("Select the experiment!")

    def appSaveModels():

        st.title('Save models')                

        with st.expander("Register model", expanded=False):

            experiments = mlflow.list_experiments()
            list_name_of_experiments = [experiments.name for experiments in experiments]
            list_name_of_experiments.insert(0, 'None')

            id_model_select = None

            col1, col2, col3 = st.columns([2,2,3])
            with col1:
                optionExperimentsName = st.selectbox(
                'Select the experiment',
                (list_name_of_experiments))
            with col2:
                if optionExperimentsName != 'None':

                    NameExperiment = optionExperimentsName
                    experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                    list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']
                    id_model_select = st.selectbox("Select the model (id_run)", list_ids_runs)


            if id_model_select != None:

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    select_X_train = st.selectbox("Select (X) train", st.session_state['dataset'])
                with col2:
                    select_y_train = st.selectbox("Select (y) train", st.session_state['dataset'])
                with col3:
                    select_X_test = st.selectbox("Select (X) test", st.session_state['dataset'])
                with col4:
                    select_y_test = st.selectbox("Select (y) test", st.session_state['dataset'])

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    targetName = st.selectbox("Input the target name", list(st.session_state[select_y_train].columns.insert(0,'None')))

                if id_model_select:
                    #st.write(id_model_select)
                    info_run = mlflow.get_run(id_model_select)

                    used_model_name = info_run.data.tags['mlflow.runName']
                    #st.write("Name Model", used_model_name)
                    used_columns = info_run.data.params['Features'].split(',')
                    #st.write("Used columns", used_columns)
                    used_target = info_run.data.params['Target']
                    #st.write("Name of target", used_target)
                    info_run.data.params.pop("Features")
                    info_run.data.params.pop("Target")
                    info_run.data.params.pop("y_pred_prob_class_1")
                    info_run.data.params.pop("y_pred_prob_class_0")
                    info_run.data.params.pop("y_test")
                    params = info_run.data.params
                    #st.write("y_test", info_run.data.params['y_test'])
                    #check_x_train = info_run.data.metrics['X train']
                    #check_x_test = info_run.data.metrics['X test']
                        
                    col1, col2 = st.columns([1,4])
                    with col1:
                        name_model_to_save = st.text_input("Input file name to save")

                    submit_save_model = st.button("Save model", key='submit_save_model')
                    if submit_save_model:
                        if name_model_to_save != '':

                            with st.spinner('Wait for it...'):
                                import scikitplot as skplt
                                
                                model = me.saveModels(have_base=None,X_train=st.session_state[select_X_train], 
                                y_train=st.session_state[select_y_train], X_test=st.session_state[select_X_test], 
                                y_test=st.session_state[select_y_test],model_name = used_model_name, columns=used_columns, 
                                target=used_target, params=params, name_model_to_save = name_model_to_save)
                                
                                metrics, y_train, y_test, y_pred, y_pred_binary = model.save()
                                #st.write(type(y_test))
                                
                                st.subheader("Output model")

                                st.write(
                                f"""
                                - **Dataset rows:** {st.session_state[select_X_train].shape[0] + st.session_state[select_X_test].shape[0]}
                                - **Dataset columns:** {st.session_state[select_X_train].shape[1] + 1}
                                - **X train rows:** {st.session_state[select_X_train].shape[0]}
                                - **X test rows:** {st.session_state[select_X_test].shape[0]}
                                - **Used columns** {used_columns}
                                - **Target name** {used_target}
                                """
                                )

                                col1, col2 = st.columns([1,1])
                                with col1:
                                    fig = px.histogram(pd.DataFrame(y_train), x=pd.DataFrame(y_train).columns[0], color=pd.DataFrame(y_train).columns[0], title='Y train')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2:
                                    fig = px.histogram(pd.DataFrame(y_test), x=pd.DataFrame(y_test).columns[0], color=pd.DataFrame(y_test).columns[0], title='Y test')
                                    fig.update_layout(bargap=0.2)
                                    st.plotly_chart(fig, use_container_width=True)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    st.table(metrics)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    probas_list = [y_pred]
                                    clf_names = [used_model_name]
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                    st.pyplot(fig)

                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                df_prob['y_pred'] = y_pred_binary
                                df_prob['prob'] = y_pred[:,1]
                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                
                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_0'})

                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_1'})
                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                st.table(analysis_prob_full)

                        else:
                            st.warning("Input file name!")

        with st.expander("Import model", expanded=False):

            uploaded_model = st.file_uploader('Upload your   model (.sav)', type='sav')

            def load_model(model):
                loaded_model = pickle.load(model)
                uploaded_model.name
                return uploaded_model.name, loaded_model

            button_load = st.button("Apply", key='button_load')
            if button_load:
                name_model, model = load_model(uploaded_model)
                pickle.dump(model, open(f'saved_models/{name_model}', 'wb'))
                st.write(
                f"""
                - **Model loaded:** {model}
                - **Model name:** {name_model}
                """
                )

    def appProduction():

        st.title('Production')

        type_of_execute = st.selectbox("Select run type",['None','Single base','Batch execution'])

        options_datasets = list(st.session_state['dataset'])
        options_datasets.insert(0, 'None')

        if type_of_execute == 'Single base':

            with st.expander("1º - Select dataset", expanded=False):

                optionDataset = st.selectbox('Select dataset', options_datasets)

            if optionDataset != 'None':
                with st.expander("2º - Select pre-processing script", expanded=False):

                    need_pre_processing = st.selectbox("Use script preprocessing?",['None','Yes','No'])

                    if need_pre_processing == 'Yes':
                        with open("libs/script_to_run_out.py", "r") as f:
                            file_contents = f.read()
                        content = st_ace(file_contents,language='python', theme='pastel_on_dark')
                    elif need_pre_processing == 'No':
                        st.info("The preprocessing script will not be executed")

                if need_pre_processing != 'None':
                    with st.expander("3º - Select model", expanded=False):

                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        options_models = st.selectbox('Select model', list_files)
                        if options_models != 'None':
                            model = pickle.load(open(f'saved_models/{options_models}', 'rb'))

            if optionDataset != 'None' and need_pre_processing != 'None' and options_models != 'None':

                st.write(
                f"""

                Information loaded
                ---------------
                - **Dataset:** {optionDataset}
                - **Pre-processing script:** {"Loaded" if need_pre_processing == 'Yes' else "Will not run"}
                - **Model**: {model}
                """
                )

                submit_prediction = st.button("Apply prediction", key='submit_prediction')
                if submit_prediction:
                    
                    if need_pre_processing == 'No':
                        with st.spinner('Wait for it...'):
                            y_pred = model.predict_proba(st.session_state[optionDataset])[:,1]

                            st.session_state[optionDataset]["prob"] = y_pred

                            breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                            labels = list(range(1,21))
                            st.session_state[optionDataset]["range_prob"] = pd.cut(st.session_state[optionDataset]["prob"], bins=breaks, labels=labels,include_lowest=True)
                            st.session_state[optionDataset]["range_prob"] = st.session_state[optionDataset]["range_prob"].astype(object)
                            st.success("Model executed successfully")

                    elif need_pre_processing == 'Yes':
                        with st.spinner('Wait for it...'):

                            import libs.script_to_run_out as script_run
                            df = script_run.run_out(base=st.session_state[optionDataset])
                            df.execute()
                            st.session_state[optionDataset] = df.base
                            del df

                            y_pred = model.predict_proba(st.session_state[optionDataset])[:,1]

                            st.session_state[optionDataset]["prob"] = y_pred

                            breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                            labels = list(range(1,21))
                            st.session_state[optionDataset]["range_prob"] = pd.cut(st.session_state[optionDataset]["prob"], bins=breaks, labels=labels,include_lowest=True)
                            st.session_state[optionDataset]["range_prob"] = st.session_state[optionDataset]["range_prob"].astype(object)
                            st.success("Model executed successfully")

        elif type_of_execute == 'Batch execution':

            name_files = []

            with st.expander("1º - Select dataset", expanded=False):

                col1, col2, col3 = st.columns([1,0.7,0.4])
                with col1:
                    path_file = st.text_input("Input the folder path")
                with col2:
                    pre_fix_name = st.text_input("File prefix")
                with col3:
                    n_files = st.number_input("Number of files", value=1, step=1,min_value=1)
                if path_file != '' and pre_fix_name != '':
                    for i in range(0+1, n_files+1):
                        name_files.append(path_file + pre_fix_name + "_" + str(i))
                    #uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, type='csv')
                    #for uploaded_file in uploaded_files:
                    #    name_files.append(uploaded_file.name)
                        #st.write("filename:", uploaded_file.name)

                    
                    st.write("**Uploaded files**")
                    st.write(name_files)
                    #st.write("**File path**")
                    #for i in name_files:
                    #    st.write(path_file + i)

            if len(name_files) > 0:
                with st.expander("2º - Select pre-processing script", expanded=False):

                    need_pre_processing = st.selectbox("Use script preprocessing?",['None','Yes','No'])

                    if need_pre_processing == 'Yes':
                        with open("libs/script_to_run_out.py", "r") as f:
                            file_contents = f.read()
                        content = st_ace(file_contents,language='python', theme='pastel_on_dark')
                    elif need_pre_processing == 'No':
                        st.info("The preprocessing script will not be executed")

                if need_pre_processing != 'None':
                    with st.expander("3º - Select model", expanded=False):

                        for _, _, arquivos in os.walk('saved_models/'):
                            list_files = arquivos
                        list_files.insert(0,'None')

                        options_models = st.selectbox('Select model', list_files)
                        if options_models != 'None':
                            model = pickle.load(open(f'saved_models/{options_models}', 'rb'))

            if len(name_files) > 0 and need_pre_processing != 'None' and options_models != 'None':

                st.write(
                f"""

                Information loaded
                ---------------
                - **Dataset:** {name_files}
                - **Pre-processing script:** {"Loaded" if need_pre_processing == 'Yes' else "Will not run"}
                - **Model**: {model}
                """
                )

                submit_prediction = st.button("Apply prediction", key='submit_prediction')
                if submit_prediction:
                    with st.spinner('Wait for it...'):

                        if need_pre_processing == 'No':

                            for file in name_files:
                                df_csv = pd.read_csv(f'{file}.csv')

                                y_pred = model.predict_proba(df_csv)[:,1]
                                df_csv["prob"] = y_pred

                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_csv["range_prob"] = pd.cut(df_csv["prob"], bins=breaks, labels=labels,include_lowest=True)
                                df_csv["range_prob"] = df_csv["range_prob"].astype(object)
                                df_csv.to_csv(f'{file}_predicted.csv', index=False)
                            del df_csv
                            st.success("Models executed successfully")

                        elif need_pre_processing == 'Yes':

                            import libs.script_to_run_out as script_run

                            for file in name_files:
                                df_csv = pd.read_csv(f'{file}.csv')

                                df = script_run.run_out(base=df_csv)
                                df.execute()
                                df_csv = df.base
                                del df

                                y_pred = model.predict_proba(df_csv)[:,1]

                                df_csv["prob"] = y_pred

                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_csv["range_prob"] = pd.cut(df_csv["prob"], bins=breaks, labels=labels,include_lowest=True)
                                df_csv["range_prob"] = df_csv["range_prob"].astype(object)
                                df_csv.to_csv(f'{file}_predicted.csv', index=False)
                            del df_csv
                            st.success("Models executed successfully")
            
    def appHyperTunning():

        st.title('Hyperparameter Tuning')

        TypeMachineLearningHyperTunning = option_menu('', ['Supervised Learning','Unsupervised Learning'], 
            default_index=0, orientation="horizontal",
            styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                    "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
        )

        if TypeMachineLearningHyperTunning == 'Supervised Learning':

            TypeClusteringModel = option_menu('', ['Grid Search','Randomized Search','Bayesian Optimization'], 
                default_index=0, orientation="horizontal",
                styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                        "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
            )

            if TypeClusteringModel == 'Grid Search':

                with st.expander("Code with hyperparameters", expanded=False):

                    button_update_code_script = st.button("Refresh page", key='button_refreshpage_code_script_gridsearch')
                    if button_update_code_script:
                        st.experimental_rerun()
                    
                    with open("libs/script_hyper_parameter.py", "r") as f:
                        file_contents = f.read()

                    content = st_ace(file_contents,language='python', theme='pastel_on_dark')

                    col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
                    with col_code_script_1:
                        opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                        if opt_code_script == 'Save script':
                            button_code_save_script = st.button("Save", key='button_code_save_script')
                            if button_code_save_script:
                                file = open("libs/script_hyper_parameter.py", "w") 
                                file.write(content)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()
                        elif opt_code_script == 'Download script':
                            st.download_button(
                                label="Download script",
                                data=file_contents,
                                file_name='script_hyper_parameter.py'
                            )
                        elif opt_code_script == 'Reset script':
                            button_code_reset_script = st.button("Reset", key='button_code_reset_script_2')
                            if button_code_reset_script:
                                with open("libs/script_hyper_parameter_backup.py", "r") as f:
                                    file_contents_backup = f.read()

                                file = open("libs/script_hyper_parameter.py", "w") 
                                file.write(file_contents_backup)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()

                with st.expander("Execute Grid Search", expanded=False):

                    col1, col2, col3, col4  = st.columns([1,1,1,1])
                    with col1:
                        use_dataset = st.selectbox("Select the dataset", st.session_state['dataset'])

                    col1, col2, col3, col4 = st.columns([1,1,1,1])
                    with col1:
                        set_scoring = st.selectbox("Select the scoring",['accuracy','f1','neg_log_loss','precision','recall','roc_auc'])
                    
                    with col2:
                        set_cv = st.number_input("Cross-validation splitting", min_value=2, max_value=100, value=2, step=1)

                    with col3:
                        experiments = mlflow.list_experiments()
                        list_name_of_experiments = [experiments.name for experiments in experiments]
                        list_name_of_experiments.insert(0, 'None')

                        id_model_select = None
                        optionExperimentsName = st.selectbox('Select the experiment',(list_name_of_experiments))

                    with col4:
                        model = None
                        if optionExperimentsName != 'None':

                            NameExperiment = optionExperimentsName
                            experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                            list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']
                            list_ids_runs.insert(0, 'None')
                            id_model_select = st.selectbox("Select the model (id_run)", list_ids_runs)

                            if id_model_select != 'None':

                                from sklearn.linear_model import LogisticRegression
                                #from catboost import CatBoostClassifier
                                from xgboost import XGBClassifier
                                from lightgbm import LGBMClassifier
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.ensemble import AdaBoostClassifier
                                from sklearn.ensemble import GradientBoostingClassifier
                                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                                from sklearn.naive_bayes import GaussianNB
                                from sklearn.tree import DecisionTreeClassifier
                                from sklearn.neighbors import KNeighborsClassifier

                                info_run = mlflow.get_run(id_model_select)
                                used_model_name = info_run.data.tags['mlflow.runName']
                                #st.write("Name Model", used_model_name)
                                used_columns = info_run.data.params['Features'].split(',')
                                #st.write("Used columns", used_columns)
                                used_target = info_run.data.params['Target']
                                #st.write("Name of target", used_target)

                                metrics = pd.DataFrame(columns=['Metric','Value'])
                                metrics.loc[0,'Metric'] = 'Accuracy'
                                metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                metrics.loc[1,'Metric'] = 'AUC'
                                metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                metrics.loc[2,'Metric'] = 'F1'
                                metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                metrics.loc[3,'Metric'] = 'Kappa'
                                metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                metrics.loc[4,'Metric'] = 'Precision'
                                metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                metrics.loc[5,'Metric'] = 'Recall'
                                metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                metrics.loc[6,'Metric'] = 'Specificity'
                                metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']

                                split = round(float(info_run.data.metrics['X test'] / st.session_state[use_dataset].shape[0]),2)

                                info_run.data.params.pop("Features")
                                info_run.data.params.pop("Target")
                                info_run.data.params.pop("y_pred_prob_class_1")
                                info_run.data.params.pop("y_pred_prob_class_0")
                                info_run.data.params.pop("y_test")
                                params = info_run.data.params
                                #st.write("Parameters", params)
                                list_params = {}
                                for k,v in params.items():
                                    if '{' in v:
                                        new_v = {}
                                        new_v[int(v.replace("{","").replace("}","").split(",")[0].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[0].strip()[-3:])
                                        new_v[int(v.replace("{","").replace("}","").split(",")[1].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[1].strip()[-3:])
                                        list_params[k] = new_v
                                    elif '.' in v:
                                        list_params[k] = float(v)
                                    elif v.isdigit():
                                        list_params[k] = int(v)
                                    elif '-' in v:
                                        if '.' in v:
                                            list_params[k] = float(v)
                                        elif 'e-' in v:
                                            list_params[k] = float(v)
                                        else:
                                            list_params[k] = int(v)
                                    elif v == 'None':
                                        list_params[k] = None
                                    elif v == 'True':
                                        list_params[k] = True
                                    elif v == 'False':
                                        list_params[k] = False
                                    else:
                                        list_params[k] = v

                                if used_model_name == 'LogisticRegression' or used_model_name == 'Tuned_LogisticRegression':
                                    model = LogisticRegression(**list_params)
                                elif used_model_name == 'XGBClassifier' or used_model_name == 'Tuned_XGBClassifier':
                                    model = XGBClassifier(**list_params)
                                elif used_model_name == 'LGBMClassifier' or used_model_name == 'Tuned_LGBMClassifier':
                                    model = LGBMClassifier(**list_params)
                                elif used_model_name == 'RandomForestClassifier' or used_model_name == 'Tuned_RandomForestClassifier':
                                    model = RandomForestClassifier(**list_params)
                                elif used_model_name == 'AdaBoostClassifier' or used_model_name == 'Tuned_AdaBoostClassifier':
                                    model = AdaBoostClassifier(**list_params)
                                elif used_model_name == 'GradientBoostingClassifier' or used_model_name == 'Tuned_GradientBoostingClassifier':
                                    model = GradientBoostingClassifier(**list_params)
                                elif used_model_name == 'LinearDiscriminantAnalysis' or used_model_name == 'Tuned_LinearDiscriminantAnalysis':
                                    model = LinearDiscriminantAnalysis(**list_params)
                                elif used_model_name == 'GaussianNB' or used_model_name == 'Tuned_GaussianNB':
                                    model = GaussianNB(**list_params)
                                elif used_model_name == 'DecisionTreeClassifier' or used_model_name == 'Tuned_DecisionTreeClassifier':
                                    model = DecisionTreeClassifier(**list_params)
                                elif used_model_name == 'KNeighborsClassifier' or used_model_name == 'Tuned_KNeighborsClassifier':
                                    model = KNeighborsClassifier(**list_params)

                    if model != None:
                        col1, col2 = st.columns([1,1])
                        with col1:
                            st.write("")
                            st.write("**Model name:**", used_model_name)
                            st.write("**Used model:**", model)
                            st.write(f"**Used columns:** {used_columns}")
                            st.write("**Target name:**", used_target)
                        with col2:
                            st.table(metrics)

                    submit_grid_tunning = st.button('Apply', key='submit_grid_tunning')

                    if submit_grid_tunning:

                        if id_model_select != 'None':

                            with st.spinner('Wait for it...'):

                                import libs.script_hyper_parameter as script_tunning
                                import scikitplot as skplt

                                list_parameters = script_tunning.run_parameters("GRID").execute()
                                #st.write(list_parameters)
                                time.sleep(0.5)

                                obj_grid_search = me.grid_search_tunning(data = st.session_state[use_dataset], target = used_target, 
                                                                            columns = used_columns, estimator = model, 
                                                                            param_grid = list_parameters, 
                                                                            scoring = set_scoring, cv = set_cv, data_split = split,
                                                                            mlflow_name_experiment = NameExperiment)
                                
                                best_score, best_params, best_model, msg, metrics, y_test, y_pred, y_pred_binary = obj_grid_search.execute_grid_search()

                                st.write("Best score")
                                st.write(best_score)
                                st.write("Best params")
                                st.write(best_params)
                                st.write("Best model")
                                st.write(best_model)

                                st.success(msg)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    st.table(metrics)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    probas_list = [y_pred]
                                    clf_names = [used_model_name]
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                    st.pyplot(fig)

                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                df_prob['y_pred'] = y_pred_binary
                                df_prob['prob'] = y_pred[:,1]
                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                
                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_0'})

                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_1'})
                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                st.table(analysis_prob_full)

                        else:

                            st.warning("Select the model!")
  

            elif TypeClusteringModel == 'Randomized Search':

                with st.expander("Code with hyperparameters", expanded=False):

                    button_update_code_script = st.button("Refresh page", key='button_refreshpage_code_script_randomsearch')
                    if button_update_code_script:
                        st.experimental_rerun()
                    
                    with open("libs/script_hyper_parameter.py", "r") as f:
                        file_contents = f.read()

                    content = st_ace(file_contents,language='python', theme='pastel_on_dark')

                    col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
                    with col_code_script_1:
                        opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                        if opt_code_script == 'Save script':
                            button_code_save_script = st.button("Save", key='button_code_save_script')
                            if button_code_save_script:
                                file = open("libs/script_hyper_parameter.py", "w") 
                                file.write(content)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()
                        elif opt_code_script == 'Download script':
                            st.download_button(
                                label="Download script",
                                data=file_contents,
                                file_name='script_hyper_parameter.py'
                            )
                        elif opt_code_script == 'Reset script':
                            button_code_reset_script = st.button("Reset", key='button_code_reset_script_2')
                            if button_code_reset_script:
                                with open("libs/script_hyper_parameter_backup.py", "r") as f:
                                    file_contents_backup = f.read()

                                file = open("libs/script_hyper_parameter.py", "w") 
                                file.write(file_contents_backup)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()

                with st.expander("Execute Randomized Search", expanded=False):

                    col1, col2, col3, col4  = st.columns([1,1,1,1])
                    with col1:
                        use_dataset = st.selectbox("Select the dataset", st.session_state['dataset'])

                    col1, col2, col3, col4, col5 = st.columns([1,0.5,0.5,1,1])
                    with col1:
                        set_scoring = st.selectbox("Select the scoring",['accuracy','f1','neg_log_loss','precision','recall','roc_auc'])
                    
                    with col2:
                        set_cv = st.number_input("Cross-validation splitting", min_value=2, max_value=100, value=2, step=1)

                    with col3:
                        set_n_iter = st.number_input("Number of iterations", min_value=2, max_value=1000, value=2, step=1)

                    with col4:
                        experiments = mlflow.list_experiments()
                        list_name_of_experiments = [experiments.name for experiments in experiments]
                        list_name_of_experiments.insert(0, 'None')

                        id_model_select = None
                        optionExperimentsName = st.selectbox('Select the experiment',(list_name_of_experiments))

                    with col5:
                        model = None
                        if optionExperimentsName != 'None':

                            NameExperiment = optionExperimentsName
                            experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                            list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']
                            list_ids_runs.insert(0, 'None')
                            id_model_select = st.selectbox("Select the model (id_run)", list_ids_runs)

                            if id_model_select != 'None':

                                from sklearn.linear_model import LogisticRegression
                                #from catboost import CatBoostClassifier
                                from xgboost import XGBClassifier
                                from lightgbm import LGBMClassifier
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.ensemble import AdaBoostClassifier
                                from sklearn.ensemble import GradientBoostingClassifier
                                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                                from sklearn.naive_bayes import GaussianNB
                                from sklearn.tree import DecisionTreeClassifier
                                from sklearn.neighbors import KNeighborsClassifier

                                info_run = mlflow.get_run(id_model_select)
                                used_model_name = info_run.data.tags['mlflow.runName']
                                #st.write("Name Model", used_model_name)
                                used_columns = info_run.data.params['Features'].split(',')
                                #st.write("Used columns", used_columns)
                                used_target = info_run.data.params['Target']
                                #st.write("Name of target", used_target)

                                metrics = pd.DataFrame(columns=['Metric','Value'])
                                metrics.loc[0,'Metric'] = 'Accuracy'
                                metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                metrics.loc[1,'Metric'] = 'AUC'
                                metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                metrics.loc[2,'Metric'] = 'F1'
                                metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                metrics.loc[3,'Metric'] = 'Kappa'
                                metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                metrics.loc[4,'Metric'] = 'Precision'
                                metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                metrics.loc[5,'Metric'] = 'Recall'
                                metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                metrics.loc[6,'Metric'] = 'Specificity'
                                metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']

                                split = round(float(info_run.data.metrics['X test'] / st.session_state[use_dataset].shape[0]),2)

                                info_run.data.params.pop("Features")
                                info_run.data.params.pop("Target")
                                info_run.data.params.pop("y_pred_prob_class_1")
                                info_run.data.params.pop("y_pred_prob_class_0")
                                info_run.data.params.pop("y_test")
                                params = info_run.data.params
                                #st.write("Parameters", params)
                                list_params = {}
                                for k,v in params.items():
                                    if '{' in v:
                                        new_v = {}
                                        new_v[int(v.replace("{","").replace("}","").split(",")[0].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[0].strip()[-3:])
                                        new_v[int(v.replace("{","").replace("}","").split(",")[1].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[1].strip()[-3:])
                                        list_params[k] = new_v
                                    elif '.' in v:
                                        list_params[k] = float(v)
                                    elif v.isdigit():
                                        list_params[k] = int(v)
                                    elif '-' in v:
                                        if '.' in v:
                                            list_params[k] = float(v)
                                        elif 'e-' in v:
                                            list_params[k] = float(v)
                                        else:
                                            list_params[k] = int(v)
                                    elif v == 'None':
                                        list_params[k] = None
                                    elif v == 'True':
                                        list_params[k] = True
                                    elif v == 'False':
                                        list_params[k] = False
                                    else:
                                        list_params[k] = v

                                if used_model_name == 'LogisticRegression' or used_model_name == 'Tuned_LogisticRegression':
                                    model = LogisticRegression(**list_params)
                                elif used_model_name == 'XGBClassifier' or used_model_name == 'Tuned_XGBClassifier':
                                    model = XGBClassifier(**list_params)
                                elif used_model_name == 'LGBMClassifier' or used_model_name == 'Tuned_LGBMClassifier':
                                    model = LGBMClassifier(**list_params)
                                elif used_model_name == 'RandomForestClassifier' or used_model_name == 'Tuned_RandomForestClassifier':
                                    model = RandomForestClassifier(**list_params)
                                elif used_model_name == 'AdaBoostClassifier' or used_model_name == 'Tuned_AdaBoostClassifier':
                                    model = AdaBoostClassifier(**list_params)
                                elif used_model_name == 'GradientBoostingClassifier' or used_model_name == 'Tuned_GradientBoostingClassifier':
                                    model = GradientBoostingClassifier(**list_params)
                                elif used_model_name == 'LinearDiscriminantAnalysis' or used_model_name == 'Tuned_LinearDiscriminantAnalysis':
                                    model = LinearDiscriminantAnalysis(**list_params)
                                elif used_model_name == 'GaussianNB' or used_model_name == 'Tuned_GaussianNB':
                                    model = GaussianNB(**list_params)
                                elif used_model_name == 'DecisionTreeClassifier' or used_model_name == 'Tuned_DecisionTreeClassifier':
                                    model = DecisionTreeClassifier(**list_params)
                                elif used_model_name == 'KNeighborsClassifier' or used_model_name == 'Tuned_KNeighborsClassifier':
                                    model = KNeighborsClassifier(**list_params)

                    if model != None:
                        col1, col2 = st.columns([1,1])
                        with col1:
                            st.write("")
                            st.write("**Model name:**", used_model_name)
                            st.write("**Used model:**", model)
                            st.write(f"**Used columns:** {used_columns}")
                            st.write("**Target name:**", used_target)
                        with col2:
                            st.table(metrics)

                    submit_random_tunning = st.button('Apply', key='submit_random_tunning')

                    if submit_random_tunning:

                        if id_model_select != 'None':

                            with st.spinner('Wait for it...'):

                                import libs.script_hyper_parameter as script_tunning
                                import scikitplot as skplt

                                list_parameters = script_tunning.run_parameters("GRID").execute()
                                #st.write(list_parameters)
                                time.sleep(0.5)

                                obj_random_search = me.random_search_tunning(data = st.session_state[use_dataset], target = used_target, 
                                                                            columns = used_columns, estimator = model, 
                                                                            param_grid = list_parameters, 
                                                                            scoring = set_scoring, cv = set_cv, n_iter = set_n_iter,
                                                                            data_split = split, mlflow_name_experiment = NameExperiment)
                                
                                best_score, best_params, best_model, msg, tot_iter, metrics, y_test, y_pred, y_pred_binary = obj_random_search.execute_random_search()

                                st.write(f"{tot_iter} iterations")
                                st.write("Best score")
                                st.write(best_score)
                                st.write("Best params")
                                st.write(best_params)
                                st.write("Best model")
                                st.write(best_model)

                                st.success(msg)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    st.table(metrics)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    probas_list = [y_pred]
                                    clf_names = [used_model_name]
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                    st.pyplot(fig)

                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                df_prob['y_pred'] = y_pred_binary
                                df_prob['prob'] = y_pred[:,1]
                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                
                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_0'})

                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_1'})
                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                st.table(analysis_prob_full)

                        else:

                            st.warning("Select the model!")

            elif TypeClusteringModel == 'Bayesian Optimization':

                with st.expander("Code with hyperparameters", expanded=False):

                    button_update_code_script = st.button("Refresh page", key='button_refreshpage_code_script_bayesian')
                    if button_update_code_script:
                        st.experimental_rerun()
                    
                    with open("libs/script_hyper_parameter_bayesian.py", "r") as f:
                        file_contents = f.read()

                    content = st_ace(file_contents,language='python', theme='pastel_on_dark')

                    col_code_script_1, col_code_script_2, col_code_script_3 = st.columns([0.6,1.2,1.2])
                    with col_code_script_1:
                        opt_code_script = st.selectbox("Select the option",['None','Save script','Reset script','Download script'])
                        if opt_code_script == 'Save script':
                            button_code_save_script = st.button("Save", key='button_code_save_script')
                            if button_code_save_script:
                                file = open("libs/script_hyper_parameter_bayesian.py", "w") 
                                file.write(content)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()
                        elif opt_code_script == 'Download script':
                            st.download_button(
                                label="Download script",
                                data=file_contents,
                                file_name='script_hyper_parameter_bayesian.py'
                            )
                        elif opt_code_script == 'Reset script':
                            button_code_reset_script = st.button("Reset", key='button_code_reset_script_2')
                            if button_code_reset_script:
                                with open("libs/script_hyper_parameter_bayesian_backup.py", "r") as f:
                                    file_contents_backup = f.read()

                                file = open("libs/script_hyper_parameter_bayesian.py", "w") 
                                file.write(file_contents_backup)
                                file.close()
                                time.sleep(1)
                                st.experimental_rerun()

                with st.expander("Execute Bayesian Optimization", expanded=False):

                    col1, col2, col3, col4  = st.columns([1,1,1,1])
                    with col1:
                        use_dataset = st.selectbox("Select the dataset", st.session_state['dataset'])

                    col1, col2, col3, col4, col5 = st.columns([1,0.5,0.5,1,1])
                    with col1:
                        set_scoring = st.selectbox("Select the scoring",['accuracy','f1','neg_log_loss','precision','recall','roc_auc'])
                    
                    with col2:
                        set_calls = st.number_input("Number of calls", min_value=2, max_value=10000, value=2, step=1)

                    with col3:
                        set_n_initial_points = st.number_input("Number of initial points", min_value=2, max_value=10000, value=2, step=1)

                    with col4:
                        experiments = mlflow.list_experiments()
                        list_name_of_experiments = [experiments.name for experiments in experiments]
                        list_name_of_experiments.insert(0, 'None')

                        id_model_select = None
                        optionExperimentsName = st.selectbox('Select the experiment',(list_name_of_experiments))

                    with col5:
                        model = None
                        if optionExperimentsName != 'None':

                            NameExperiment = optionExperimentsName
                            experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                            list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']
                            list_ids_runs.insert(0, 'None')
                            id_model_select = st.selectbox("Select the model (id_run)", list_ids_runs)

                            if id_model_select != 'None':

                                from sklearn.linear_model import LogisticRegression
                                #from catboost import CatBoostClassifier
                                from xgboost import XGBClassifier
                                from lightgbm import LGBMClassifier
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.ensemble import AdaBoostClassifier
                                from sklearn.ensemble import GradientBoostingClassifier
                                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                                from sklearn.naive_bayes import GaussianNB
                                from sklearn.tree import DecisionTreeClassifier
                                from sklearn.neighbors import KNeighborsClassifier

                                info_run = mlflow.get_run(id_model_select)
                                used_model_name = info_run.data.tags['mlflow.runName']
                                #st.write("Name Model", used_model_name)
                                used_columns = info_run.data.params['Features'].split(',')
                                #st.write("Used columns", used_columns)
                                used_target = info_run.data.params['Target']
                                #st.write("Name of target", used_target)

                                metrics = pd.DataFrame(columns=['Metric','Value'])
                                metrics.loc[0,'Metric'] = 'Accuracy'
                                metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                                metrics.loc[1,'Metric'] = 'AUC'
                                metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                                metrics.loc[2,'Metric'] = 'F1'
                                metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                                metrics.loc[3,'Metric'] = 'Kappa'
                                metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                                metrics.loc[4,'Metric'] = 'Precision'
                                metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                                metrics.loc[5,'Metric'] = 'Recall'
                                metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                                metrics.loc[6,'Metric'] = 'Specificity'
                                metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']

                                split = round(float(info_run.data.metrics['X test'] / st.session_state[use_dataset].shape[0]),2)

                                info_run.data.params.pop("Features")
                                info_run.data.params.pop("Target")
                                info_run.data.params.pop("y_pred_prob_class_1")
                                info_run.data.params.pop("y_pred_prob_class_0")
                                info_run.data.params.pop("y_test")
                                params = info_run.data.params
                                #st.write("Parameters", params)
                                list_params = {}
                                for k,v in params.items():
                                    if '{' in v:
                                        new_v = {}
                                        new_v[int(v.replace("{","").replace("}","").split(",")[0].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[0].strip()[-3:])
                                        new_v[int(v.replace("{","").replace("}","").split(",")[1].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[1].strip()[-3:])
                                        list_params[k] = new_v
                                    elif '.' in v:
                                        list_params[k] = float(v)
                                    elif v.isdigit():
                                        list_params[k] = int(v)
                                    elif '-' in v:
                                        if '.' in v:
                                            list_params[k] = float(v)
                                        elif 'e-' in v:
                                            list_params[k] = float(v)
                                        else:
                                            list_params[k] = int(v)
                                    elif v == 'None':
                                        list_params[k] = None
                                    elif v == 'True':
                                        list_params[k] = True
                                    elif v == 'False':
                                        list_params[k] = False
                                    else:
                                        list_params[k] = v

                                if used_model_name == 'LogisticRegression' or used_model_name == 'Tuned_LogisticRegression':
                                    model = LogisticRegression(**list_params)
                                elif used_model_name == 'XGBClassifier' or used_model_name == 'Tuned_XGBClassifier':
                                    model = XGBClassifier(**list_params)
                                elif used_model_name == 'LGBMClassifier' or used_model_name == 'Tuned_LGBMClassifier':
                                    model = LGBMClassifier(**list_params)
                                elif used_model_name == 'RandomForestClassifier' or used_model_name == 'Tuned_RandomForestClassifier':
                                    model = RandomForestClassifier(**list_params)
                                elif used_model_name == 'AdaBoostClassifier' or used_model_name == 'Tuned_AdaBoostClassifier':
                                    model = AdaBoostClassifier(**list_params)
                                elif used_model_name == 'GradientBoostingClassifier' or used_model_name == 'Tuned_GradientBoostingClassifier':
                                    model = GradientBoostingClassifier(**list_params)
                                elif used_model_name == 'LinearDiscriminantAnalysis' or used_model_name == 'Tuned_LinearDiscriminantAnalysis':
                                    model = LinearDiscriminantAnalysis(**list_params)
                                elif used_model_name == 'GaussianNB' or used_model_name == 'Tuned_GaussianNB':
                                    model = GaussianNB(**list_params)
                                elif used_model_name == 'DecisionTreeClassifier' or used_model_name == 'Tuned_DecisionTreeClassifier':
                                    model = DecisionTreeClassifier(**list_params)
                                elif used_model_name == 'KNeighborsClassifier' or used_model_name == 'Tuned_KNeighborsClassifier':
                                    model = KNeighborsClassifier(**list_params)

                    if model != None:
                        col1, col2 = st.columns([1,1])
                        with col1:
                            st.write("")
                            st.write("**Model name:**", used_model_name)
                            st.write("**Used model:**", model)
                            st.write(f"**Used columns:** {used_columns}")
                            st.write("**Target name:**", used_target)
                        with col2:
                            st.table(metrics)

                    submit_bayesian_tunning = st.button('Apply', key='submit_bayesian_tunning')

                    if submit_bayesian_tunning:

                        if id_model_select != 'None':

                            with st.spinner('Wait for it...'):

                                import libs.script_hyper_parameter_bayesian as script_tunning
                                import scikitplot as skplt

                                list_parameters, name_parameters = script_tunning.run_parameters("Bayesian").execute()
                                #st.write(list_parameters)
                                time.sleep(0.5)

                                obj_bayesian = me.bayesian_tunning(data = st.session_state[use_dataset], target = used_target, 
                                                                    columns = used_columns, estimator = model, 
                                                                    space = list_parameters, name_parameters = name_parameters,
                                                                    scoring = set_scoring, data_split = split, set_calls = set_calls,
                                                                    set_n_initial_points = set_n_initial_points,                                                                    
                                                                    mlflow_name_experiment = NameExperiment)
                                
                                best_score, bayesian_params, best_model, msg, metrics, y_test, y_pred, y_pred_binary = obj_bayesian.execute_bayesian()

                                st.write("Best score")
                                st.write(best_score)
                                st.write("Best params")
                                st.write(bayesian_params)
                                st.write("Best model")
                                st.write(best_model)

                                image = Image.open('files_output/result_bayesian.png')
                                st.image(image)

                                st.success(msg)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    st.table(metrics)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=False, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_confusion_matrix(y_test, y_pred_binary, normalize=True, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_roc(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_ks_statistic(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_precision_recall(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)

                                col1, col2, col3 = st.columns([1,1,1])
                                with col1:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_cumulative_gain(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col2:
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_lift_curve(y_test, y_pred, ax=ax)
                                    st.pyplot(fig)
                                with col3:
                                    probas_list = [y_pred]
                                    clf_names = [used_model_name]
                                    fig, ax = plt.subplots()
                                    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, ax=ax)
                                    st.pyplot(fig)

                                df_prob = pd.DataFrame(y_test, columns=[used_target])
                                df_prob['y_pred'] = y_pred_binary
                                df_prob['prob'] = y_pred[:,1]
                                breaks = [round(num,2) for num in list(np.linspace(0,1,21))]
                                labels = list(range(1,21))
                                df_prob["range_prob"] = pd.cut(df_prob["prob"], bins=breaks, labels=labels)
                                
                                analysis_prob1 = df_prob[df_prob[used_target] == 0].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_0'})

                                analysis_prob2 = df_prob[df_prob[used_target] == 1].groupby("range_prob", as_index=False)\
                                                                                    .agg({used_target:'count'})\
                                                                                    .rename(columns={used_target:'class_1'})
                                analysis_prob_full = pd.concat([analysis_prob1, analysis_prob2["class_1"]],axis=1)
                                analysis_prob_full['range_prob'] = analysis_prob_full['range_prob'].astype(int)
                                analysis_prob_full["total_customers"] = analysis_prob_full["class_0"] + analysis_prob_full["class_1"]
                                analysis_prob_full['rate_class_1'] = analysis_prob_full['class_1'] / analysis_prob_full['total_customers']
                                st.table(analysis_prob_full)

        elif TypeMachineLearningHyperTunning == 'Unsupervised Learning':

            st.warning('Coming soon!')  

    def appModelValidation():

        st.title('Model validation')

        type_validation = st.multiselect("Select validation type", ['None','K-Fold Cross-validation'])

        if 'K-Fold Cross-validation' in type_validation:

            with st.expander("K-Fold Cross-validation", expanded=False):

                model = None

                col1, col2, col3, col4 = st.columns([0.75,0.375,0.375,1.5])
                with col1:
                    dataset_kfold = st.selectbox("Select dataset", st.session_state['dataset'])
                with col2:
                    n_fold_kfold = st.number_input("Folder number", step=1, value=1, min_value=1)
                with col3:
                    shuffle_kfold = st.selectbox("Shuffle before?",['Yes','No'])
                    var_shuffle_kfold = True if shuffle_kfold == 'Yes' else False                        

                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1:
                    experiments = mlflow.list_experiments()
                    list_name_of_experiments = [experiments.name for experiments in experiments]
                    list_name_of_experiments.insert(0, 'None')

                    id_model_select = None
                    optionExperimentsName = st.selectbox('Select the experiment',(list_name_of_experiments))
                with col2:
                    if optionExperimentsName != 'None':

                        from sklearn.linear_model import LogisticRegression
                        #from catboost import CatBoostClassifier
                        from xgboost import XGBClassifier
                        from lightgbm import LGBMClassifier
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.ensemble import AdaBoostClassifier
                        from sklearn.ensemble import GradientBoostingClassifier
                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                        from sklearn.naive_bayes import GaussianNB
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.neighbors import KNeighborsClassifier

                        NameExperiment = optionExperimentsName
                        experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                        list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']
                        id_model_select = st.selectbox("Select the model (id_run)", list_ids_runs)

                        info_run = mlflow.get_run(id_model_select)
                        used_model_name = info_run.data.tags['mlflow.runName']
                        #st.write("Name Model", used_model_name)
                        used_columns = info_run.data.params['Features'].split(',')
                        #st.write("Used columns", used_columns)
                        used_target = info_run.data.params['Target']
                        #st.write("Name of target", used_target)

                        metrics = pd.DataFrame(columns=['Metric','Value'])
                        metrics.loc[0,'Metric'] = 'Accuracy'
                        metrics.loc[0,'Value'] = info_run.data.metrics['Accuracy']
                        metrics.loc[1,'Metric'] = 'AUC'
                        metrics.loc[1,'Value'] = info_run.data.metrics['AUC']
                        metrics.loc[2,'Metric'] = 'F1'
                        metrics.loc[2,'Value'] = info_run.data.metrics['F1']
                        metrics.loc[3,'Metric'] = 'Kappa'
                        metrics.loc[3,'Value'] = info_run.data.metrics['Kappa']
                        metrics.loc[4,'Metric'] = 'Precision'
                        metrics.loc[4,'Value'] = info_run.data.metrics['Precision']
                        metrics.loc[5,'Metric'] = 'Recall'
                        metrics.loc[5,'Value'] = info_run.data.metrics['Recall']
                        metrics.loc[6,'Metric'] = 'Specificity'
                        metrics.loc[6,'Value'] = info_run.data.metrics['Specificity']

                        info_run.data.params.pop("Features")
                        info_run.data.params.pop("Target")
                        info_run.data.params.pop("y_pred_prob_class_1")
                        info_run.data.params.pop("y_pred_prob_class_0")
                        info_run.data.params.pop("y_test")
                        params = info_run.data.params
                        #st.write("Parameters", params)
                        list_params = {}
                        for k,v in params.items():
                            if '{' in v:
                                        new_v = {}
                                        new_v[int(v.replace("{","").replace("}","").split(",")[0].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[0].strip()[-3:])
                                        new_v[int(v.replace("{","").replace("}","").split(",")[1].strip()[0])] = float(v.replace("{","").replace("}","").split(",")[1].strip()[-3:])
                                        list_params[k] = new_v
                            elif '.' in v:
                                list_params[k] = float(v)
                            elif v.isdigit():
                                list_params[k] = int(v)
                            elif '-' in v:
                                if '.' in v:
                                    list_params[k] = float(v)
                                elif 'e-' in v:
                                    list_params[k] = float(v)
                                else:
                                    list_params[k] = int(v)
                            elif v == 'None':
                                list_params[k] = None
                            elif v == 'True':
                                list_params[k] = True
                            elif v == 'False':
                                list_params[k] = False
                            else:
                                list_params[k] = v

                        #if used_model_name == 'LogisticRegression':
                        #    model = LogisticRegression(**list_params)
                        #elif used_model_name == 'XGBClassifier':
                        #    model = XGBClassifier(**list_params)
                        #elif used_model_name == 'LGBMClassifier':
                        #    model = LGBMClassifier(**list_params)
                        #elif used_model_name == 'RandomForestClassifier':
                        #    model = RandomForestClassifier(**list_params)
                        #elif used_model_name == 'AdaBoostClassifier':
                        #    model = AdaBoostClassifier(**list_params)
                        #elif used_model_name == 'GradientBoostingClassifier':
                        #    model = GradientBoostingClassifier(**list_params)
                        #elif used_model_name == 'LinearDiscriminantAnalysis':
                        #    model = LinearDiscriminantAnalysis(**list_params)
                        #elif used_model_name == 'GaussianNB':
                        #    model = GaussianNB(**list_params)
                        #elif used_model_name == 'DecisionTreeClassifier':
                        #    model = DecisionTreeClassifier(**list_params)
                        #elif used_model_name == 'KNeighborsClassifier':
                        #    model = KNeighborsClassifier(**list_params)

                        if used_model_name == 'LogisticRegression' or used_model_name == 'Tuned_LogisticRegression':
                            model = LogisticRegression(**list_params)
                        elif used_model_name == 'XGBClassifier' or used_model_name == 'Tuned_XGBClassifier':
                            model = XGBClassifier(**list_params)
                        elif used_model_name == 'LGBMClassifier' or used_model_name == 'Tuned_LGBMClassifier':
                            model = LGBMClassifier(**list_params)
                        elif used_model_name == 'RandomForestClassifier' or used_model_name == 'Tuned_RandomForestClassifier':
                            model = RandomForestClassifier(**list_params)
                        elif used_model_name == 'AdaBoostClassifier' or used_model_name == 'Tuned_AdaBoostClassifier':
                            model = AdaBoostClassifier(**list_params)
                        elif used_model_name == 'GradientBoostingClassifier' or used_model_name == 'Tuned_GradientBoostingClassifier':
                            model = GradientBoostingClassifier(**list_params)
                        elif used_model_name == 'LinearDiscriminantAnalysis' or used_model_name == 'Tuned_LinearDiscriminantAnalysis':
                            model = LinearDiscriminantAnalysis(**list_params)
                        elif used_model_name == 'GaussianNB' or used_model_name == 'Tuned_GaussianNB':
                            model = GaussianNB(**list_params)
                        elif used_model_name == 'DecisionTreeClassifier' or used_model_name == 'Tuned_DecisionTreeClassifier':
                            model = DecisionTreeClassifier(**list_params)
                        elif used_model_name == 'KNeighborsClassifier' or used_model_name == 'Tuned_KNeighborsClassifier':
                            model = KNeighborsClassifier(**list_params)

                if model != None:
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.write("")
                        st.write("**Used model:**", model)
                        st.write(f"**Used columns:** {used_columns}")
                        st.write("**Target name:**", used_target)
                    with col2:
                        st.table(metrics)
                        

                submit_cv_kfold = st.button("Apply", key='submit_cv_kfold')
                if submit_cv_kfold:
                    with st.spinner('Wait for it...'):
                        from sklearn.model_selection import cross_validate
                        from sklearn.metrics import make_scorer
                        from sklearn.model_selection import KFold
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score

                        scorings = {
                            'accuracy': make_scorer(accuracy_score),
                            'sensitivity': make_scorer(recall_score),
                            'specificity': make_scorer(recall_score,pos_label=0),
                            'precision': make_scorer(precision_score),
                            'F1': make_scorer(f1_score),
                            'roc_auc': make_scorer(roc_auc_score)
                        }

                        X = st.session_state[dataset_kfold][used_columns]
                        y = st.session_state[dataset_kfold][[used_target]]

                        cv = KFold(n_splits=n_fold_kfold, random_state=42, shuffle=var_shuffle_kfold)
                        models_scorings = cross_validate(model, X, y, cv=cv, scoring = scorings)
                        df_scorings = pd.DataFrame(models_scorings).drop(['fit_time','score_time'],axis=1)
                        df_scorings.reset_index(inplace=True)
                        df_scorings.columns = ['Fold', 'test_accuracy', 'test_sensitivity', 'test_specificity',
                                            'test_precision', 'test_F1', 'test_roc_auc']
                        df_scorings['Fold'] = df_scorings['Fold'] + 1

                        st.subheader("Results")

                        col1, col2 = st.columns([0.8,0.2])
                        with col1:
                            import plotly.express as px
                            fig = px.line(df_scorings,x='Fold', y=list(df_scorings.columns))#, markers=True)
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            df_mean = pd.DataFrame(df_scorings.drop(['Fold'],axis=1).mean(), columns=['Mean'])
                            df_mean.reset_index(inplace=True)
                            df_mean.columns = ['Metric', 'Mean']
                            st.table(df_mean)





    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Models Experiments':
        appModelsExperiments()

    elif appSelectionSubCat == 'Report Experiments':
        appReportExperiments()

    elif appSelectionSubCat == 'Hyperparameter Tuning':
        appHyperTunning()

    elif appSelectionSubCat == 'Model validation':
        appModelValidation()

    elif appSelectionSubCat == 'Save models':
        appSaveModels()

    elif appSelectionSubCat == 'Production':
        appProduction()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/image7.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Model Engeneering
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )