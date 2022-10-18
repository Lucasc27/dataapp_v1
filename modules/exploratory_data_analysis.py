from typing import Any
import streamlit as st
from PIL import Image
from pandas_profiling import ProfileReport
import sweetviz
import time
import datetime as dt
from datetime import date
import pandas as pd
import numpy as np
import libs.EDA_graphs as EDA
import sys
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_option_menu import option_menu


def app():

    appSelectionSubCat = option_menu('Select option', ['Home','Data Reports','Summary and Statistics','Special Charts'], 
        icons=['house'], 
        menu_icon="bi bi-box-arrow-in-right", default_index=0, orientation="horizontal",
        styles={"container": {"padding": "1!important", "background-color": "#F9F7F7"},
                                "nav-link": {"font-size": "13px","--hover-color": "#eee"}}
    )

    #appSelectionSubCat = st.sidebar.selectbox('Submenu',['Home','Data Reports','Summary and Statistics'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def page_EDA_profile_and_sweetviz():

        st.title("Reports")

        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            col1, col2, col3 = st.columns([1,1,2])
            #pr = None
            with col1:

                with st.form(key='my_form_pandas_profile', clear_on_submit=True):

                    st.write("Create report using pandas profile")
                    minimum_report = st.selectbox("Minimum report?",['Yes', 'No'])
                    submit = st.form_submit_button('Submit')
                    from streamlit_pandas_profiling import st_profile_report
                    def report_profile_min(df):
                        dt_today = date.today()
                        mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]
                        profile = ProfileReport(df, minimal=True, infer_dtypes=False)
                        profile.to_file(output_file=f'files_output/pandas_profile_output_mininum_{mes_ano}.html')

                    def report_profile(df):
                        dt_today = date.today()
                        mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]
                        profile = ProfileReport(df, infer_dtypes=False)
                        #pr = df.profile_report()
                        #return st_profile_report(pr)
                        #return st.write("TESTE")
                        profile.to_file(output_file=f'files_output/pandas_profile_output_{mes_ano}.html')

                    if submit:
                        if minimum_report == 'Yes':

                            start_time_func = time.time()
                            report_profile_min(st.session_state[optionDataset])
                            end_time_func = time.time()
                            finished_time = float(round(end_time_func - start_time_func,2))

                            st.success(f"Created successfully, done in {finished_time} seconds")
                            time.sleep(1)
                            st.experimental_rerun()

                        elif minimum_report == 'No':

                            start_time_func = time.time()
                            report_profile(st.session_state[optionDataset])
                            end_time_func = time.time()
                            finished_time = float(round(end_time_func - start_time_func,2))

                            st.success(f"Created successfully, done in {finished_time} seconds")
                            time.sleep(1)
                            st.experimental_rerun()

            with col2:

                with st.form(key='my_form_sweetviz', clear_on_submit=True):

                    st.write("Create report using SweetViz")
                    name_target = st.text_input("Target variable")
                    submit = st.form_submit_button('Submit')

                    if submit:
                        start_time_func = time.time()

                        dt_today = date.today()
                        mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]
                        my_report  = sweetviz.analyze([st.session_state[optionDataset],'Exploratory'], target_feat=name_target)
                        my_report.show_html(f'files_output/sweetviz_output_{mes_ano}.html', open_browser=True)

                        end_time_func = time.time()
                        finished_time = float(round(end_time_func - start_time_func,2))

                        st.success(f"Created successfully, done in {str(finished_time)} seconds")
                        time.sleep(1)
                        st.experimental_rerun()

            with col3:

                with st.form(key='my_form_sweetviz_compare', clear_on_submit=True):


                    st.write("Create report comparison with different bases using SweetViz")
                    optionDatasetCompare = st.selectbox('Select the dataset for compare',(st.session_state['dataset']))
                    have_target = st.selectbox("Have target variable", ['','No','Yes'])
                    name_target = st.text_input("Target variable name")
                    submit = st.form_submit_button('Submit')

                    if submit:
                        if optionDatasetCompare != optionDataset:
                            if have_target == 'Yes':
                                if name_target:
                                    start_time_func = time.time()

                                    dt_today = date.today()
                                    mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]

                                    df_1 = st.session_state[optionDataset].drop([name_target],axis=1)
                                    df_1['flag_compare'] = 0

                                    df_2 = st.session_state[optionDatasetCompare]
                                    df_2['flag_compare'] = 1

                                    df_fim = pd.concat([df_1, df_2],axis=0)
                                        
                                    my_report  = sweetviz.compare_intra(df_fim, df_fim["flag_compare"]==1, [optionDataset, optionDatasetCompare])

                                    #my_report  = sweetviz.analyze([st.session_state[optionDataset],'Exploratory'], target_feat=name_target)
                                    my_report.show_html(f'files_output/sweetviz_output_compare_{mes_ano}.html', open_browser=True)

                                    del df_1['flag_compare']
                                    del df_2['flag_compare']

                                    end_time_func = time.time()
                                    finished_time = float(round(end_time_func - start_time_func,2))

                                    st.success(f"Created successfully, done in {str(finished_time)} seconds")
                                    time.sleep(2)
                                    st.experimental_rerun()
                                else:
                                    st.error("Input the target variable!")
                                    time.sleep(1)
                                    st.experimental_rerun()

                            elif have_target == 'No':
                                
                                start_time_func = time.time()

                                dt_today = date.today()
                                mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]

                                df_1 = st.session_state[optionDataset]
                                df_1['flag_compare'] = 0

                                df_2 = st.session_state[optionDatasetCompare]
                                df_2['flag_compare'] = 1

                                df_fim = pd.concat([df_1, df_2],axis=0)
                                    
                                my_report  = sweetviz.compare_intra(df_fim, df_fim["flag_compare"]==1, [optionDatasetCompare, optionDataset])

                                #my_report  = sweetviz.analyze([st.session_state[optionDataset],'Exploratory'], target_feat=name_target)
                                my_report.show_html(f'files_output/sweetviz_output_compare_{mes_ano}.html', open_browser=True)

                                del df_1['flag_compare']
                                del df_2['flag_compare']

                                end_time_func = time.time()
                                finished_time = float(round(end_time_func - start_time_func,2))

                                st.success(f"Created successfully, done in {str(finished_time)} seconds")
                                time.sleep(2)
                                st.experimental_rerun()
                            
                            else:

                                st.error("Check the option if there is a target variable!")
                                time.sleep(1)
                                st.experimental_rerun()
                                
                        else:
                            
                            if have_target == 'Yes':
                                if name_target:
                                    start_time_func = time.time()

                                    dt_today = date.today()
                                    mes_ano = str(dt_today).split('-')[2] + "_" + str(dt_today).split('-')[1] + "_" + str(dt_today).split('-')[0]

                                    my_report  = sweetviz.compare_intra(st.session_state[optionDataset], st.session_state[optionDataset][name_target]==1, ["Target = 1", "Target = 0"])

                                    my_report.show_html(f'files_output/sweetviz_output_compare_{mes_ano}.html', open_browser=True)

                                    end_time_func = time.time()
                                    finished_time = float(round(end_time_func - start_time_func,2))

                                    st.success(f"Created successfully, done in {str(finished_time)} seconds")
                                    time.sleep(2)
                                    st.experimental_rerun()

                                    st.error("It's not allowed to compare the same dataset!")
                                    time.sleep(1)
                                    st.experimental_rerun()

                                else:
                                    st.error("Input the target variable!")
                                    time.sleep(1)
                                    st.experimental_rerun()
                            else:
                                st.error("Check the option if there is a target variable!")
                                time.sleep(1)
                                st.experimental_rerun()

            #pr = st.session_state[optionDataset].profile_report(infer_dtypes=False)
            #st_profile_report(pr)

        else:
            st.write("There is no Dataset loaded")

    def page_EDA_summary_statistics():  

        st.title("Summary and statistics") 

        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            numeric_features = st.session_state[optionDataset].select_dtypes(include=[np.number]).columns
            categorical_features = st.session_state[optionDataset].select_dtypes(include=[np.object,'category']).columns

            #st.sidebar.write(numeric_features)
            #st.sidebar.write(categorical_features)

            eda_plot = EDA.EDA(st.session_state[optionDataset])

            def basic_info(df):
                #st.header("Data")
                st.write('Number of observations', df.shape[0]) 
                st.write('Number of variables', df.shape[1])
                st.write('Number total of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))

            #Visualize data
            basic_info(st.session_state[optionDataset])

            options = ["Statistic descriptive", "Statistic univariate", "Statistic multivariate"]
            menu = st.multiselect("Select type of analysis:", options)
            
            if 'Statistic descriptive' in menu:
                with st.expander("Statistic descriptive", expanded=False):

                    st.header("Statistic descriptive")

                    #@st.cache
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

                        return pd.DataFrame({'Objects Types': x_dtypes, 'Size': df_sizeof['Size'].values, 'NaN': df.isna().sum(), 'NaN%': round((df.isna().sum()/len(df))*100,2), 'Unique':df.nunique()})

                    #@st.cache
                    def get_stats(df):
                        stats_num = df.describe()
                        if df.select_dtypes([np.object,'category']).empty :
                            return stats_num.transpose(), None
                        if df.select_dtypes(np.number).empty :
                            return None, df.describe(include=[np.object,'category']).transpose()
                        else:
                            return stats_num.transpose(), df.describe(include=[np.object,'category']).transpose()

                    #Data statistics
                    df_info = get_info(st.session_state[optionDataset])   
                    df_stat_num, df_stat_obj = get_stats(st.session_state[optionDataset])

                    st.markdown('**Numerical summary**')
                    st.dataframe(df_stat_num)
                    st.markdown('**Categorical summary**')
                    st.dataframe(df_stat_obj)
                    st.markdown('**Missing Values**')
                    st.dataframe(df_info)

            if 'Statistic univariate' in menu:
                with st.expander("Statistic univariate", expanded=False):
                    
                    @st.cache
                    def pf_of_info(df,col):

                        x_dtypes = pd.DataFrame({'Columns':list(df.columns),'type':df.dtypes.values})
                        x_dtypes['type_str'] = x_dtypes['type'].apply(lambda x: str(x))

                        info = dict()
                        info['Type'] =  str(x_dtypes[x_dtypes["Columns"] == col[0]]["type_str"].values[0])
                        info['Unique'] = df[col].nunique()
                        info['n_zeros'] = (len(df) - np.count_nonzero(df[col]))
                        info['p_zeros'] = round(info['n_zeros'] * 100 / len(df),2)
                        info['nan'] = df[col].isna().sum()
                        info['p_nan'] =  (df[col].isna().sum() / df.shape[0]) * 100
                        return pd.DataFrame(info, index = col).T.round(2)
                    
                    @st.cache     
                    def pd_of_stats(df,col):
                        #Descriptive Statistics
                        stats = dict()
                        stats['Mean']  = df[col].mean()
                        stats['Std']   = df[col].std()
                        stats['Var'] = df[col].var()
                        stats['Kurtosis'] = df[col].kurtosis()
                        stats['Skewness'] = df[col].skew()
                        stats['Coefficient Variance'] = stats['Std'] / stats['Mean']
                        return pd.DataFrame(stats, index = col).T.round(2)

                    @st.cache     
                    def pd_of_stats_quantile(df,col):
                        df_no_na = df[col].dropna()
                        stats_q = dict()

                        stats_q['Min'] = df[col].min()
                        label = {0.25:"Q1", 0.5:'Median', 0.75:"Q3"}
                        for percentile in np.array([0.25, 0.5, 0.75]):
                            stats_q[label[percentile]] = df_no_na.quantile(percentile)
                        stats_q['Max'] = df[col].max()
                        stats_q['Range'] = stats_q['Max']-stats_q['Min']
                        stats_q['IQR'] = stats_q['Q3']-stats_q['Q1']
                        return pd.DataFrame(stats_q, index = col).T.round(2)   

                    def plot_univariate(df, obj_plot, main_var, radio_plot_uni):
                        if 'Histogram' in radio_plot_uni:
                            st.subheader('Histogram')
                            bins, range_ = None, None
                            hue_opt = st.selectbox("Hue (categorical) *optional",obj_plot.columns.insert(0,None))
                            bins_ = st.slider('Number of bins *optional', value = 10, key='bins_histogram')
                            range_ = st.slider('Choose range optional', float(obj_plot.df[main_var].min()), \
                                float(obj_plot.df[main_var].max()),(float(obj_plot.df[main_var].min()),float(obj_plot.df[main_var].max())))    
                            #button_histogram = st.button('Plot histogram chart', key='histogram_plot_univariate')
                            #if button_histogram:
                            fig = obj_plot.histogram_num(main_var, hue_opt, bins_, range_)
                            fig.update_layout(autosize=False, height=800)
                            st.plotly_chart(fig,use_container_width=False)

                        if 'BoxPlot' in radio_plot_uni:
                            st.subheader('Boxplot')
                            # col_x, hue_opt = None, None
                            col_x  = st.selectbox("Choose x variable (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot')
                            hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot')
                            #button_boxplot = st.button('Plot boxplot chart', key='boxplot_plot_univariate')
                            #if button_boxplot:
                            fig = obj_plot.box_plot(main_var,col_x, hue_opt)
                            fig.update_layout(autosize=False, height=800)
                            st.plotly_chart(fig,use_container_width=False)

                        if 'Distribution Plot' in radio_plot_uni:
                            st.subheader('Distribution Plot')
                            #fig, ax = plt.subplots()
                            #obj_plot.DistPlot(main_var)
                            #fig = ff.create_distplot(df[main_var], group_labels=['teste'], bin_size=.5, curve_type='normal')
                            bins, range_ = None, None
                            bins_distplot = st.slider('Number of values in inside the bins *optional', value = 10, key='bins_distplot')
                            fig = ff.create_distplot([df[main_var]], [str(main_var)], bin_size=bins_distplot, curve_type='normal')
                            fig.update_layout(autosize=False, height=900)
                            st.plotly_chart(fig, use_container_width=False)

                    st.header("Statistic univariate")
                    st.markdown("Summary statistics of only one variable in the dataset.")
                    main_var = st.selectbox("Choose one variable to analyze:", list(st.session_state[optionDataset].columns.insert(0,None)))

                    if main_var in numeric_features:
                        if main_var != None:
                            st.subheader("Variable info")
                            st.table(pf_of_info(st.session_state[optionDataset], [main_var]).T)
                            st.subheader("Descriptive Statistics")
                            st.table((pd_of_stats(st.session_state[optionDataset], [main_var])).T)
                            st.subheader("Quantile Statistics") 
                            st.table((pd_of_stats_quantile(st.session_state[optionDataset], [main_var])).T) 
                            
                            chart_univariate = st.multiselect('Charts', ['None','Histogram', 'BoxPlot', 'Distribution Plot'])
                            
                            plot_univariate(st.session_state[optionDataset], eda_plot, main_var, chart_univariate)

                    if main_var in categorical_features:
                        st.table(st.session_state[optionDataset][main_var].describe(include = [np.object, 'category']))
                        st.bar_chart(st.session_state[optionDataset][main_var].value_counts().to_frame())

            if 'Statistic multivariate' in menu:

                with st.expander("Statistic multivariate", expanded=False):

                    def plot_multivariate(obj_plot, radio_plot):
                        
                        if 'Boxplot' in radio_plot:
                            st.subheader('Boxplot')
                            col_y  = st.selectbox("Choose main variable (numerical)",obj_plot.num_vars.insert(0,None), key ='boxplot_multivariate')
                            col_x  = st.selectbox("Choose x variable (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                            hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                            #if st.sidebar.button('Plot boxplot chart'):
                            show_boxplot = st.checkbox("Show me boxplot")
                            if show_boxplot:
                                st.plotly_chart(obj_plot.box_plot(col_y,col_x, hue_opt))
                        
                        #if radio_plot == ('Violin'):
                        #    st.subheader('Violin')
                        #    col_y  = st.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='violin_multivariate')
                        #    col_x  = st.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='violin_multivariate')
                        #    hue_opt = st.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='violin_multivariate')
                        #    split = st.checkbox("Split",key='violin_multivariate')
                        #    #if st.sidebar.button('Plot violin chart'):
                        #    fig, ax = plt.subplots()
                        #    #fig = px.violin(st.session_state[optionDataset], y=col_y)
                        #    #st.plotly_chart(fig)
                        #    obj_plot.violin(col_y,col_x, hue_opt, split)
                        #    st.pyplot(fig)
                        
                        #if radio_plot == ('Swarmplot'):
                        #    st.subheader('Swarmplot')
                        #    col_y = st.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='swarmplot_multivariate')
                        #    col_x = st.selectbox("Choose x variable (categorical) *optional", obj_plot.columns.insert(0,None),key='swarmplot_multivariate')
                        #    hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None),key='swarmplot_multivariate')
                        #    split = st.checkbox("Split", key ='swarmplot_multivariate')
                        #   #if st.sidebar.button('Plot swarmplot chart'):
                        #    fig = obj_plot.swarmplot(col_y,col_x, hue_opt, split)
                        #    st.pyplot()

                        def pretty(method):
                            return method.capitalize()
                    
                        if 'Correlation' in radio_plot:

                            st.subheader('Heatmap Correlation Plot')
                            correlation = st.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman','phik', 'cramer v'), format_func=pretty)
                            cols_list = st.multiselect("Select columns",obj_plot.columns)
                            st.markdown("If none selected, it will plot the correlation of all numeric variables.")
                            cols_list_rv = st.multiselect("Select columns to remove from chart",obj_plot.columns)
                            #if st.sidebar.button('Plot heatmap chart'):
                            fig, ax = plt.subplots()
                            obj_plot.Corr(cols_list, correlation, cols_list_rv)
                            show_corr = st.checkbox("Show me correlation")
                            if show_corr:
                                st.pyplot(fig)

                        #def map_func(function):
                        #    dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
                        #    return dic[function]
                        
                        #if radio_plot == ('Heatmap'):
                        #    st.subheader('Heatmap between vars')
                        #    st.markdown(" In order to plot this chart remember that the order of the selection matters, \
                        #        chooose in order the variables that will build the pivot table: row, column and value.")
                        #    cols_list = st.multiselect("Select 3 variables (2 categorical and 1 numeric)",obj_plot.columns, key= 'heatmapvars_multivariate')
                        #    agg_func = st.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func)
                        #    #if st.sidebar.button('Plot heatmap between vars'):
                        #    fig = obj_plot.heatmap_vars(cols_list, agg_func)
                        #    st.pyplot()
                        
                        if 'Histogram' in radio_plot:
                            st.subheader('Histogram')
                            col_hist = st.selectbox("Choose main variable", obj_plot.num_vars.insert(0,None), key = 'hist')
                            hue_opt = st.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
                            bins_, range_ = None, None
                            bins_ = st.slider('Number of bins optional', value = 30)
                            if col_hist != None:
                                range_ = st.slider('Choose range optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                                        (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
                            #if st.button('Plot histogram chart'):
                            show_hist = st.checkbox("Show me histogram")
                            if show_hist:
                                st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

                        if 'Scatterplot' in radio_plot: 
                            st.subheader('Scatter plot')
                            col_x = st.selectbox("Choose X variable (numerical)", obj_plot.num_vars.insert(0,None), key = 'scatter')
                            col_y = st.selectbox("Choose Y variable (numerical)", obj_plot.num_vars.insert(0,None), key = 'scatter')
                            hue_opt = st.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'scatter')
                            size_opt = st.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'scatter')
                            #if st.sidebar.button('Plot scatter chart'):
                            show_scatterplot = st.checkbox("Show me scatterplot")
                            if show_scatterplot:
                                st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))

                        if 'Countplot' in radio_plot:
                            st.subheader('Count Plot')
                            col_count_plot = st.selectbox("Choose main variable",obj_plot.columns.insert(0,None), key = 'countplot')
                            hue_opt = st.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'countplot')
                            #if st.sidebar.button('Plot Countplot'):
                            fig, ax = plt.subplots()
                            obj_plot.CountPlot(col_count_plot, hue_opt)
                            show_countplot = st.checkbox("Show me countplot")
                            if show_countplot:
                                st.pyplot(fig)
                        
                        if 'Barplot' in radio_plot:
                            st.subheader('Barplot') 
                            col_y = st.selectbox("Choose main variable (numerical)",obj_plot.num_vars.insert(0,None), key='barplot')
                            col_x = st.selectbox("Choose x variable (categorical)", obj_plot.columns.insert(0,None),key='barplot')
                            hue_opt = st.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0,None),key='barplot')
                            #if st.sidebar.button('Plot barplot chart'):
                            show_barplot = st.checkbox("Show me barplot")
                            if show_barplot:
                                st.plotly_chart(obj_plot.bar_plot(col_y, col_x, hue_opt))
                            #st.plotly_chart(obj_plot.bar_plot(col_y, hue_opt))

                        if 'Lineplot' in radio_plot:
                            st.subheader('Lineplot') 
                            col_y = st.selectbox("Choose main variable (numerical)",obj_plot.num_vars.insert(0,None), key='lineplot')
                            col_x = st.selectbox("Choose x variable (categorical)", obj_plot.columns.insert(0,None),key='lineplot')
                            hue_opt = st.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
                            group = st.selectbox("Group color (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
                            #if st.sidebar.button('Plot lineplot chart'):
                            show_lineplot = st.checkbox("Show me lineplot")
                            if show_lineplot:
                                st.plotly_chart(obj_plot.line_plot(col_y,col_x, hue_opt, group))

                    st.header("Statistic multivariate")

                    st.markdown('Here you can visualize your data by choosing one of the chart options available!')
                    st.subheader('Data visualization options')
                    radio_plot = st.multiselect('Choose plot style', ['Boxplot', 'Correlation', 'Histogram', \
                        'Scatterplot', 'Countplot', 'Barplot', 'Lineplot'])

                    plot_multivariate(eda_plot, radio_plot)

                if 'Correlation' in radio_plot:

                    with st.expander("Correlations information", expanded=False):

                        image = Image.open('images/correlations.png')
                        st.image(image, caption='Correlations table')

                        image2 = Image.open('images/correlations_2.png')
                        st.image(image2, caption='Person, cramer and phik')
        else:
            st.write("There is no Dataset loaded")

    def page_EDA_special_charts():

        st.title("Special Charts")

        with st.expander("Compare scatter plot", expanded=False):
            
            col1_compare_scatterplot_0, col2_compare_scatterplot_0 = st.columns([0.15,0.85])
            with col1_compare_scatterplot_0:
                compare = st.selectbox("Compare", ['Two graphs','Three graphs'])

            if compare == 'Two graphs':
                col1_compare_scatterplot_1, col2_compare_scatterplot_1 = st.columns([1,1])
                with col1_compare_scatterplot_1:
                    select_dataset_1 = st.selectbox("Select the first dataset to compare", st.session_state['dataset'])
                with col2_compare_scatterplot_1:
                    select_dataset_2 = st.selectbox("Select the second dataset to compare", st.session_state['dataset'])

            elif compare == 'Three graphs':
                col1_compare_scatterplot_1, col2_compare_scatterplot_1, col3_compare_scatterplot_1 = st.columns([1,1,1])
                with col1_compare_scatterplot_1:
                    select_dataset_1 = st.selectbox("Select the first dataset to compare", st.session_state['dataset'])
                with col2_compare_scatterplot_1:
                    select_dataset_2 = st.selectbox("Select the second dataset to compare", st.session_state['dataset'])
                with col3_compare_scatterplot_1:
                    select_dataset_3 = st.selectbox("Select the third dataset to compare", st.session_state['dataset'])

            if compare == 'Two graphs':
                col1_compare_scatterplot_2, col2_compare_scatterplot_2 = st.columns([1,1])
                with col1_compare_scatterplot_2:
                    x_var_dataset_1 = st.selectbox("Select the X axis variable in first dataset", st.session_state[select_dataset_1].columns)
                    y_var_dataset_1 = st.selectbox("Select the Y axis variable in first dataset", st.session_state[select_dataset_1].columns)
                    var_color_dataset_1 = st.selectbox("Select color division in first dataset", st.session_state[select_dataset_1].columns)
                with col2_compare_scatterplot_2:
                    x_var_dataset_2 = st.selectbox("Select the X axis variable in second dataset", st.session_state[select_dataset_2].columns)
                    y_var_dataset_2 = st.selectbox("Select the Y axis variable in second dataset", st.session_state[select_dataset_2].columns)
                    var_color_dataset_2 = st.selectbox("Select color division in second dataset", st.session_state[select_dataset_2].columns)
                

            elif compare == 'Three graphs':
                col1_compare_scatterplot_2, col2_compare_scatterplot_2, col3_compare_scatterplot_2 = st.columns([1,1,1])
                with col1_compare_scatterplot_2:
                    x_var_dataset_1 = st.selectbox("Select the X axis variable in first dataset", st.session_state[select_dataset_1].columns)
                    y_var_dataset_1 = st.selectbox("Select the Y axis variable in first dataset", st.session_state[select_dataset_1].columns)
                    var_color_dataset_1 = st.selectbox("Select color division in first dataset", st.session_state[select_dataset_1].columns)
                with col2_compare_scatterplot_2:
                    x_var_dataset_2 = st.selectbox("Select the X axis variable in second dataset", st.session_state[select_dataset_2].columns)
                    y_var_dataset_2 = st.selectbox("Select the Y axis variable in second dataset", st.session_state[select_dataset_2].columns)
                    var_color_dataset_2 = st.selectbox("Select color division in second dataset", st.session_state[select_dataset_2].columns)
                with col3_compare_scatterplot_2:
                    x_var_dataset_3 = st.selectbox("Select the X axis variable in third dataset", st.session_state[select_dataset_3].columns)
                    y_var_dataset_3 = st.selectbox("Select the Y axis variable in third dataset", st.session_state[select_dataset_3].columns)
                    var_color_dataset_3 = st.selectbox("Select color division in third dataset", st.session_state[select_dataset_3].columns)

            col1_color_palette_1, col2_color_palette_1, col3_color_palette_1 = st.columns([0.2,0.2,1])
            with col1_color_palette_1:
                color_palette_1 = st.color_picker('Color to min value', '#00FF00')
            with col2_color_palette_1:
                color_palette_2 = st.color_picker('Color to max value', '#FF0000')

            show_me_graph_compare = st.checkbox("Show me", key='show_me_graph_compare')
            if show_me_graph_compare:
                if compare == 'Two graphs':
                    col1_graph_compare_scatterplot, col2_graph_compare_scatterplot = st.columns([1,1])
                    with col1_graph_compare_scatterplot:
                        fig = px.scatter(st.session_state[select_dataset_1], x=x_var_dataset_1, y=y_var_dataset_1, color=var_color_dataset_1, color_continuous_scale=[(0,color_palette_1), (1, color_palette_2)])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2_graph_compare_scatterplot:
                        fig = px.scatter(st.session_state[select_dataset_2], x=x_var_dataset_2, y=y_var_dataset_2, color=var_color_dataset_2, color_continuous_scale=[(0,color_palette_1), (1, color_palette_2)])
                        st.plotly_chart(fig, use_container_width=True)
                
                elif compare == 'Three graphs':
                    col1_graph_compare_scatterplot, col2_graph_compare_scatterplot, col3_graph_compare_scatterplot = st.columns([1,1,1])
                    with col1_graph_compare_scatterplot:
                        fig = px.scatter(st.session_state[select_dataset_1], x=x_var_dataset_1, y=y_var_dataset_1, color=var_color_dataset_1, color_continuous_scale=[(0,color_palette_1), (1, color_palette_2)])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2_graph_compare_scatterplot:
                        fig = px.scatter(st.session_state[select_dataset_2], x=x_var_dataset_2, y=y_var_dataset_2, color=var_color_dataset_2, color_continuous_scale=[(0,color_palette_1), (1, color_palette_2)])
                        st.plotly_chart(fig, use_container_width=True)
                    with col3_graph_compare_scatterplot:
                        fig = px.scatter(st.session_state[select_dataset_3], x=x_var_dataset_3, y=y_var_dataset_3, color=var_color_dataset_3, color_continuous_scale=[(0,color_palette_1), (1, color_palette_2)])
                        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Visualize missing values (NaN)", expanded=False):

            optionDataset = st.selectbox(
            'Select the dataset',
            (st.session_state['dataset']), key='optionDataset_missing')

            if optionDataset:

                col1, col2 = st.columns([0.2,0.8])
                with col1:
                    select_or_all_vars_missing_value_graph = st.selectbox("Select variables?", ['None','Yes','No, show me all'])
                with col2:
                    if select_or_all_vars_missing_value_graph == 'Yes':
                        select_vars_missing_value_graph = st.multiselect("Select the variables", st.session_state[optionDataset].columns)

                button_missing_value_graph = st.button("Apply", key='button_missing_value_graph')
                if button_missing_value_graph:
                    if select_or_all_vars_missing_value_graph != 'None':
                        with st.spinner('Wait for it...'):
                            #col1, col2 = st.columns([1,1])
                            #with col1:
                            #    width = st.slider("plot width", 1, 25, 3)
                            #with col2:
                            #    height = st.slider("plot height", 1, 25, 1)
                            if select_or_all_vars_missing_value_graph == 'Yes':
                                fig, ax = plt.subplots(figsize=(20, 5))
                                ax = sns.heatmap(st.session_state[optionDataset][select_vars_missing_value_graph].isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
                                ax.axes.set_title("Missing values", fontsize = 20)
                                st.pyplot(fig)
                            elif select_or_all_vars_missing_value_graph == 'No, show me all':
                                fig, ax = plt.subplots(figsize=(20, 5))
                                ax = sns.heatmap(st.session_state[optionDataset].isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
                                ax.axes.set_title("Missing values", fontsize = 20)
                                st.pyplot(fig)
                    else:
                        st.warning("Insert option!")

            else:
                st.write("There is no Dataset loaded")

        with st.expander("Visualize binary values (1 and 0)", expanded=False):

            optionDataset = st.selectbox(
            'Select the dataset',
            (st.session_state['dataset']), key='optionDataset_binary')

            if optionDataset:

                list_vars_binary = []
                for col in st.session_state[optionDataset].columns:
                    if len(st.session_state[optionDataset][col].value_counts()) <= 2:
                        list_values = list(st.session_state[optionDataset][col].value_counts().index)
                        if 0 in list_values or 1 in list_values:
                            list_vars_binary.append(col)

                st.write(
                f"""
                - **Variables:** {", ".join(list_vars_binary)}
                """
                )
                st.write("")

                if len(list_vars_binary) > 0:

                    col1, col2 = st.columns([0.2,0.8])
                    with col1:
                        select_or_all_vars_binary_value_graph = st.selectbox("Select variables?", ['None','Yes','No, show me all'], key='select_or_all_vars_binary_value_graph')
                    with col2:
                        if select_or_all_vars_binary_value_graph == 'Yes':
                            select_vars_binary_value_graph = st.multiselect("Select the variables", st.session_state[optionDataset][list_vars_binary].columns, key='select_vars_binary_value_graph')

                    button_binary_value_graph = st.button("Apply", key='button_binary_value_graph')
                    if button_binary_value_graph:
                        if select_or_all_vars_binary_value_graph != 'None':
                            with st.spinner('Wait for it...'):
                                #col1, col2 = st.columns([1,1])
                                #with col1:
                                #    width = st.slider("plot width", 1, 25, 3)
                                #with col2:
                                #    height = st.slider("plot height", 1, 25, 1)
                                st.info("Values ​​equal to 1 are in yellow")
                                #col1, col2 = st.columns([0.7,0.3])
                                #with col1:
                                if select_or_all_vars_binary_value_graph == 'Yes':
                                    fig, ax = plt.subplots()
                                    ax = sns.heatmap(st.session_state[optionDataset][select_vars_binary_value_graph] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    ax.axes.set_title("Binary values", fontsize = 20)
                                    st.pyplot(fig)                                    
                                elif select_or_all_vars_binary_value_graph == 'No, show me all':
                                    fig, ax = plt.subplots()
                                    ax = sns.heatmap(st.session_state[optionDataset][list_vars_binary] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    ax.axes.set_title("Binary values", fontsize = 20)
                                    st.pyplot(fig)  
                        else:
                            st.warning("Insert option!")
                else:
                    st.write("There is no binary variable")

            else:
                st.write("There is no Dataset loaded")

        with st.expander("Visualize categorical count of values", expanded=False):

            optionDataset = st.selectbox(
            'Select the dataset',
            (st.session_state['dataset']), key='optionDataset_category')

            if optionDataset:

                col1, col2 = st.columns([0.2,0.8])
                with col1:
                    select_max_category_number = st.number_input("Maximum number of unique values", value=1, step=1, min_value=1)

                list_vars_category = []
                for col in st.session_state[optionDataset].columns:
                    if len(st.session_state[optionDataset][col].value_counts()) <= select_max_category_number:
                        #list_values = list(st.session_state[optionDataset][col].value_counts().index)
                        #if 0 in list_values or 1 in list_values:
                        list_vars_category.append(col)

                st.write(
                f"""
                - **Variables:** {", ".join(list_vars_category)}
                """
                )
                st.write("")

                if len(list_vars_category) > 0:

                    col1, col2 = st.columns([0.2,0.8])
                    with col1:
                        select_or_all_vars_category_value_graph = st.selectbox("Select variables?", ['None','Yes','No, show me all'], key='select_or_all_vars_category_value_graph')
                    with col2:
                        if select_or_all_vars_category_value_graph == 'Yes':
                            select_vars_category_value_graph = st.multiselect("Select the variables", st.session_state[optionDataset][list_vars_category].columns, key='select_vars_category_value_graph')

                    button_category_value_graph = st.button("Apply", key='button_category_value_graph')
                    if button_category_value_graph:
                        if select_or_all_vars_category_value_graph != 'None':
                            with st.spinner('Wait for it...'):
                                #col1, col2 = st.columns([1,1])
                                #with col1:
                                #    width = st.slider("plot width", 1, 25, 3)
                                #with col2:
                                #    height = st.slider("plot height", 1, 25, 1)
                                #st.info("Values equal to 1 are in yellow")
                                #col1, col2 = st.columns([0.7,0.3])
                                #with col1:
                                if select_or_all_vars_category_value_graph == 'Yes':
                                    counts = st.session_state[optionDataset][select_vars_category_value_graph].apply(pd.value_counts)
                                    sns.heatmap(counts)
                                    st.pyplot()
                                    #fig, ax = plt.subplots()
                                    #ax = sns.heatmap(st.session_state[optionDataset][select_vars_category_value_graph] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    #ax.axes.set_title("Binary values", fontsize = 20)
                                                                        
                                elif select_or_all_vars_category_value_graph == 'No, show me all':
                                    counts = st.session_state[optionDataset][list_vars_category].apply(pd.value_counts)
                                    sns.heatmap(counts)
                                    st.pyplot()
                                    #fig, ax = plt.subplots()
                                    #ax = sns.heatmap(st.session_state[optionDataset][list_vars_binary] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    #ax.axes.set_title("Binary values", fontsize = 20)
                                        
                        else:
                            st.warning("Insert option!")
                else:
                    st.write("There is no categorical variable")

        with st.expander("Compare distribution with violin chart", expanded=False):

            optionDataset = st.selectbox(
            'Select the dataset',
            (st.session_state['dataset']), key='optionDataset_violinplot')

            if optionDataset:

                cols_numerical_violinplot = list(st.session_state[optionDataset].select_dtypes(include=[np.number]).columns)

                st.write(
                f"""
                - **Numeric variables:** {", ".join(cols_numerical_violinplot)}
                """
                )
                st.write("")

                if len(cols_numerical_violinplot) > 0:

                    col1, col2 = st.columns([0.2,0.8])
                    with col1:
                        select_or_all_vars_violinplot = st.selectbox("Select variables?", ['None','Yes','No, show me all'], key='select_or_all_vars_violinplot')
                    with col2:
                        if select_or_all_vars_violinplot == 'Yes':
                            select_vars_violinplot = st.multiselect("Select the variables", st.session_state[optionDataset][cols_numerical_violinplot].columns, key='select_vars_violinplot')
                    
                    col1, col2 = st.columns([0.2,0.8])
                    with col1:
                        select_target_violinplot = st.selectbox("Select the target variable", st.session_state[optionDataset].columns)

                    button_violinplot = st.button("Apply", key='button_violinplot')
                    if button_violinplot:
                        if select_or_all_vars_violinplot != 'None':
                            with st.spinner('Wait for it...'):
                                #col1, col2 = st.columns([1,1])
                                #with col1:
                                #    width = st.slider("plot width", 1, 25, 3)
                                #with col2:
                                #    height = st.slider("plot height", 1, 25, 1)
                                #st.info("Values equal to 1 are in yellow")
                                #col1, col2 = st.columns([0.7,0.3])
                                #with col1:
                                if select_or_all_vars_violinplot == 'Yes':
                                    if not select_target_violinplot in select_vars_violinplot:
                                        select_vars_violinplot.append(select_target_violinplot)
                                    #st.write(select_vars_violinplot)
                                    from sklearn import preprocessing
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit_transform(st.session_state[optionDataset][select_vars_violinplot].drop(select_target_violinplot,axis=1))
                                    df_std = pd.DataFrame(df_std, columns=st.session_state[optionDataset][select_vars_violinplot].drop(select_target_violinplot,axis=1).columns)
                                    df_std = pd.concat([df_std, st.session_state[optionDataset][select_target_violinplot]],axis=1)

                                    df_melt = pd.melt(df_std, id_vars=select_target_violinplot, var_name="Variable", value_name="Values")
                                    sns.violinplot(x="Variable", y="Values", hue=select_target_violinplot, data=df_melt, split=True)
                                    plt.xticks(rotation = 90)
                                    st.pyplot()
                                    #fig, ax = plt.subplots()
                                    #ax = sns.heatmap(st.session_state[optionDataset][select_vars_category_value_graph] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    #ax.axes.set_title("Binary values", fontsize = 20)
                                                                        
                                elif select_or_all_vars_violinplot == 'No, show me all':
                                    if not select_target_violinplot in cols_numerical_violinplot:
                                        cols_numerical_violinplot.append(select_target_violinplot)
                                    #st.write(cols_numerical_violinplot)
                                    from sklearn import preprocessing
                                    scaler = preprocessing.StandardScaler()
                                    df_std = scaler.fit_transform(st.session_state[optionDataset][cols_numerical_violinplot].drop(select_target_violinplot,axis=1))
                                    df_std = pd.DataFrame(df_std, columns=st.session_state[optionDataset][cols_numerical_violinplot].drop(select_target_violinplot,axis=1).columns)
                                    df_std = pd.concat([df_std, st.session_state[optionDataset][select_target_violinplot]],axis=1)

                                    df_melt = pd.melt(df_std, id_vars=select_target_violinplot, var_name="Variable", value_name="Values")
                                    sns.violinplot(x="Variable", y="Values", hue=select_target_violinplot, data=df_melt, split=True)
                                    plt.xticks(rotation = 90)
                                    st.pyplot()
                                    #fig, ax = plt.subplots()
                                    #ax = sns.heatmap(st.session_state[optionDataset][list_vars_binary] == 1, yticklabels = False, cbar = False, cmap = 'viridis')
                                    #ax.axes.set_title("Binary values", fontsize = 20)
                                        
                        else:
                            st.warning("Insert option!")
                else:
                    st.write("There is no numeric variable")

        with st.expander("Variable stability", expanded=False):

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                optionDataset = st.selectbox(
                'Select the dataset',
                (st.session_state['dataset']), key='optionDataset_stability')

            cols_stability_chart = list(st.session_state[optionDataset].columns)
            cols_stability_chart.insert(0, 'None')

            st.write("---------------")

            if optionDataset:
                
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    var_dt = st.selectbox("Input the date variable", cols_stability_chart)
                with col2:
                    var_y = st.selectbox("Input the y-axis variable", cols_stability_chart)
                with col3:
                    var_color = st.selectbox("Input the category variable", cols_stability_chart)                        

                show_graph_stability_1 = st.checkbox("Show graph")
                if show_graph_stability_1:
                    if var_dt != 'None':
                        if var_y != 'None':
                            if var_color == 'None':
                                df_stability_1 = pd.DataFrame(st.session_state[optionDataset].groupby([var_dt], as_index=False)[var_y].sum())
                                df_stability_1[var_dt] =  pd.to_datetime(df_stability_1[var_dt], format="%Y-%m-%d")
                                df_stability_1['year'] = df_stability_1[var_dt].dt.year
                                df_stability_1['month'] = df_stability_1[var_dt].dt.month
                                df_stability_1['day'] = df_stability_1[var_dt].dt.day

                                format = 'MMM DD, YYYY'
                                month_day_min = df_stability_1[df_stability_1['year'] == df_stability_1['year'].min()]
                                day_min = month_day_min[month_day_min['month'] == month_day_min['month'].min()]['day'].min()
                                month_day_max = df_stability_1[df_stability_1['year'] == df_stability_1['year'].max()]
                                day_max = month_day_max[month_day_max['month'] == month_day_max['month'].max()]['day'].max()
                                start_date = dt.date(year=df_stability_1['year'].min(),month=df_stability_1[df_stability_1['year'] == df_stability_1['year'].min() ]['month'].min(),day=day_min)
                                end_date = dt.date(year=df_stability_1['year'].max(),month=df_stability_1[df_stability_1['year'] == df_stability_1['year'].max() ]['month'].max(),day=day_max)
                                max_days = end_date-start_date

                                col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
                                with col2:
                                    range_dt = st.slider("",start_date,end_date, (start_date,end_date), format=format)

                                fig = px.line(df_stability_1, x=var_dt, y=var_y, range_x=[range_dt[0],range_dt[1]])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                df_stability_1 = pd.DataFrame(st.session_state[optionDataset].groupby([var_dt, var_color], as_index=False)[var_y].sum())
                                df_stability_1[var_dt] =  pd.to_datetime(df_stability_1[var_dt], format="%Y-%m-%d")
                                df_stability_1['year'] = df_stability_1[var_dt].dt.year
                                df_stability_1['month'] = df_stability_1[var_dt].dt.month
                                df_stability_1['day'] = df_stability_1[var_dt].dt.day

                                format = 'MMM DD, YYYY'
                                month_day_min = df_stability_1[df_stability_1['year'] == df_stability_1['year'].min()]
                                day_min = month_day_min[month_day_min['month'] == month_day_min['month'].min()]['day'].min()
                                month_day_max = df_stability_1[df_stability_1['year'] == df_stability_1['year'].max()]
                                day_max = month_day_max[month_day_max['month'] == month_day_max['month'].max()]['day'].max()
                                start_date = dt.date(year=df_stability_1['year'].min(),month=df_stability_1[df_stability_1['year'] == df_stability_1['year'].min() ]['month'].min(),day=day_min)
                                end_date = dt.date(year=df_stability_1['year'].max(),month=df_stability_1[df_stability_1['year'] == df_stability_1['year'].max() ]['month'].max(),day=day_max)
                                max_days = end_date-start_date

                                col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
                                with col2:
                                    range_dt = st.slider("",start_date,end_date, (start_date,end_date), format=format)

                                fig = px.line(df_stability_1, x=var_dt, y=var_y, color=var_color, range_x=[range_dt[0],range_dt[1]])
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insert the y-axis variable!")
                    else:
                        st.warning("Insert the date variable!")




            else:
                st.warning("There is no Dataset loaded")

    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Data Reports':
        page_EDA_profile_and_sweetviz()

    elif appSelectionSubCat == 'Summary and Statistics':
        page_EDA_summary_statistics()

    elif appSelectionSubCat == 'Special Charts':
        page_EDA_special_charts()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/image5.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Exploratory Data Analysis (EDA) :mag:
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )