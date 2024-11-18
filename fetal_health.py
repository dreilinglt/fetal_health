# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title and Description
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', )
st.write('Utilize our advanced Machine Learning applitcation to predict fetal health classifications.')

# Read csv file to show example dataframe
fetal_df = pd.read_csv('fetal_health.csv')
fetal_df_trimmed = fetal_df.drop(columns='fetal_health').head()
fetal_df_rounded = fetal_df_trimmed.round({'baseline value':0, 'accelerations':3,'fetal_movement':0,'uterine_contractions':3,'light_decelerations':3,'severe_decelerations':3,'prolongued_decelerations':3,'abnormal_short_term_variability':1,'mean_value_of_short_term_variability':1,'percentage_of_time_with_abnormal_long_term_variability':1,'mean_value_of_long_term_variability':1,'histogram_width':1,'histogram_min':1,'histogram_max':1,'histogram_number_of_peaks':1,'histogram_number_of_zeroes':1,'histogram_mode':1,'histogram_mean':1,'histogram_median':1,'histogram_variance':1,'histogram_tendency':1})
# display_df = fetal_df_trimmed.style.hide(axis='index')
# Sidebar inputs
st.sidebar.header("Fetal Health Features Input")
user_file = st.sidebar.file_uploader('Upload your data', help='File must be in CSV format')
st.sidebar.warning(':warning: Ensure your data strictly follows the format outlined below')
st.sidebar.write(fetal_df_rounded)
selection = st.sidebar.radio('Choose the model you wish to use:',options=['Random Forest','Decision Tree','AdaBoost','Soft Voting'])
st.sidebar.info(f':heavy_check_mark: You selected: {selection}')

# Load Pickles
dt_pickle = open('fetal_decision_tree.pickle', 'rb') 
dt_clf = pickle.load(dt_pickle) 
dt_pickle.close()

rf_pickle = open('fetal_random_forest.pickle', 'rb') 
rf_clf = pickle.load(rf_pickle) 
rf_pickle.close()

ada_pickle = open('fetal_ada_boost.pickle', 'rb') 
ada_clf = pickle.load(ada_pickle) 
ada_pickle.close()

sv_pickle = open('fetal_soft_voting.pickle', 'rb') 
sv_clf = pickle.load(sv_pickle) 
sv_pickle.close()

# Create a mapping dictionary
mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}

# Create a function to color cells based on values
def highlight(value):
    if value == 'Normal':
        return 'background-color: lime'
    elif value == 'Suspect':
        return 'background-color: yellow'
    elif value == 'Pathological':
        return 'background-color: orange'
    else:
        return ''  # No styling for other values

if user_file is not None:
    st.success(':white_check_mark: CSV file uploaded successfully')
    user_df = pd.read_csv(user_file)
    # Predict Health Classification
    if selection == 'Decision Tree':
        st.header('Predicting Fetal Health Class Using Decision Tree Model')
        dt_new_prediction = dt_clf.predict(user_df)
        # Create a dataframe to show the predictions
        user_df['Predicted Fetal Health'] = dt_new_prediction
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map(mapping)
        # Apply styling with color coding and format the display
        colored_df = user_df.style.applymap(highlight, subset=['Predicted Fetal Health']).format({
        'baseline value': '{:.0f}', 
        'accelerations': '{:.3f}', 
        'fetal_movement': '{:.3f}', 
        'uterine_contractions': '{:.3f}', 
        'light_decelerations': '{:.3f}', 
        'severe_decelerations': '{:.1f}', 
        'prolongued_decelerations': '{:.3f}', 
        'abnormal_short_term_variability': '{:.1f}', 
        'mean_value_of_short_term_variability': '{:.1f}', 
        'percentage_of_time_with_abnormal_long_term_variability': '{:.1f}', 
        'mean_value_of_long_term_variability': '{:.1f}', 
        'histogram_width': '{:.1f}', 
        'histogram_min': '{:.1f}', 
        'histogram_max': '{:.1f}', 
        'histogram_number_of_peaks': '{:.1f}', 
        'histogram_number_of_zeroes': '{:.1f}', 
        'histogram_mode': '{:.1f}', 
        'histogram_mean': '{:.1f}', 
        'histogram_median': '{:.1f}', 
        'histogram_variance': '{:.1f}', 
        'histogram_tendency': '{:.1f}', })
        st.write(colored_df)
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Feature Importance
        with tab1:
            st.write("### Feature Importance")
            st.image('fetal_dt_feature_imp.svg')
            st.caption("Visualization of the Most Important Features.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetal_dt_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetal_class_dt_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    elif selection == 'Random Forest':
        st.header('Predicting Fetal Health Class Using Random Forest Model')
        rf_new_prediction = rf_clf.predict(user_df)
        rf_prediction_prob = rf_clf.predict_proba(user_df).max(axis=1)*100
        # Create a dataframe to show the predictions
        user_df['Predicted Fetal Health'] = rf_new_prediction
        user_df['Prediction Probability'] = rf_prediction_prob
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map(mapping)
        # Apply styling with color coding and format the display
        colored_df = user_df.style.applymap(highlight, subset=['Predicted Fetal Health']).format({
        'baseline value': '{:.0f}', 
        'accelerations': '{:.3f}', 
        'fetal_movement': '{:.3f}', 
        'uterine_contractions': '{:.3f}', 
        'light_decelerations': '{:.3f}', 
        'severe_decelerations': '{:.1f}', 
        'prolongued_decelerations': '{:.3f}', 
        'abnormal_short_term_variability': '{:.1f}', 
        'mean_value_of_short_term_variability': '{:.1f}', 
        'percentage_of_time_with_abnormal_long_term_variability': '{:.1f}', 
        'mean_value_of_long_term_variability': '{:.1f}', 
        'histogram_width': '{:.1f}', 
        'histogram_min': '{:.1f}', 
        'histogram_max': '{:.1f}', 
        'histogram_number_of_peaks': '{:.1f}', 
        'histogram_number_of_zeroes': '{:.1f}', 
        'histogram_mode': '{:.1f}', 
        'histogram_mean': '{:.1f}', 
        'histogram_median': '{:.1f}', 
        'histogram_variance': '{:.1f}', 
        'histogram_tendency': '{:.1f}', 
        'Prediction Probability': '{:.1f}'})
        st.write(colored_df)
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Feature Importance
        with tab1:
            st.write("### Feature Importance")
            st.image('fetal_rf_feature_imp.svg')
            st.caption("Visualization of the Most Important Features.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetal_rf_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetal_class_rf_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Greens').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    elif selection == 'AdaBoost':
        st.header('Predicting Fetal Health Class Using AdaBoost Model')
        ada_new_prediction = ada_clf.predict(user_df)
        ada_prediction_prob = ada_clf.predict_proba(user_df).max(axis=1)*100
        # Create a dataframe to show the predictions
        user_df['Predicted Fetal Health'] = ada_new_prediction
        user_df['Prediction Probability'] = ada_prediction_prob
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map(mapping)
        # Apply styling with color coding and format the display
        colored_df = user_df.style.applymap(highlight, subset=['Predicted Fetal Health']).format({
        'baseline value': '{:.0f}', 
        'accelerations': '{:.3f}', 
        'fetal_movement': '{:.3f}', 
        'uterine_contractions': '{:.3f}', 
        'light_decelerations': '{:.3f}', 
        'severe_decelerations': '{:.1f}', 
        'prolongued_decelerations': '{:.3f}', 
        'abnormal_short_term_variability': '{:.1f}', 
        'mean_value_of_short_term_variability': '{:.1f}', 
        'percentage_of_time_with_abnormal_long_term_variability': '{:.1f}', 
        'mean_value_of_long_term_variability': '{:.1f}', 
        'histogram_width': '{:.1f}', 
        'histogram_min': '{:.1f}', 
        'histogram_max': '{:.1f}', 
        'histogram_number_of_peaks': '{:.1f}', 
        'histogram_number_of_zeroes': '{:.1f}', 
        'histogram_mode': '{:.1f}', 
        'histogram_mean': '{:.1f}', 
        'histogram_median': '{:.1f}', 
        'histogram_variance': '{:.1f}', 
        'histogram_tendency': '{:.1f}', 
        'Prediction Probability': '{:.1f}'})
        st.write(colored_df)
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Feature Importance
        with tab1:
            st.write("### Feature Importance")
            st.image('fetal_ada_feature_imp.svg')
            st.caption("Visualization of the Most Important Features.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetal_ada_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetal_class_ada_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Reds').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    elif selection == 'Soft Voting':
        st.header('Predicting Fetal Health Class Using Soft Voting Model')
        sv_new_prediction = sv_clf.predict(user_df)
        sv_prediction_prob = sv_clf.predict_proba(user_df).max(axis=1)*100
        sv_new_prediction = sv_clf.predict(user_df)
        sv_prediction_prob = sv_clf.predict_proba(user_df).max(axis=1)*100
        # Create a dataframe to show the predictions
        user_df['Predicted Fetal Health'] = sv_new_prediction
        user_df['Prediction Probability'] = sv_prediction_prob
        user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map(mapping)
        # Apply styling with color coding and format the display
        colored_df = user_df.style.applymap(highlight, subset=['Predicted Fetal Health']).format({
        'baseline value': '{:.0f}', 
        'accelerations': '{:.3f}', 
        'fetal_movement': '{:.3f}', 
        'uterine_contractions': '{:.3f}', 
        'light_decelerations': '{:.3f}', 
        'severe_decelerations': '{:.1f}', 
        'prolongued_decelerations': '{:.3f}', 
        'abnormal_short_term_variability': '{:.1f}', 
        'mean_value_of_short_term_variability': '{:.1f}', 
        'percentage_of_time_with_abnormal_long_term_variability': '{:.1f}', 
        'mean_value_of_long_term_variability': '{:.1f}', 
        'histogram_width': '{:.1f}', 
        'histogram_min': '{:.1f}', 
        'histogram_max': '{:.1f}', 
        'histogram_number_of_peaks': '{:.1f}', 
        'histogram_number_of_zeroes': '{:.1f}', 
        'histogram_mode': '{:.1f}', 
        'histogram_mean': '{:.1f}', 
        'histogram_median': '{:.1f}', 
        'histogram_variance': '{:.1f}', 
        'histogram_tendency': '{:.1f}', 
        'Prediction Probability': '{:.1f}'})
        st.write(colored_df)
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Feature Importance
        with tab1:
            st.write("### Feature Importance")
            st.image('fetal_sv_feature_imp.svg')
            st.caption("Visualization of the Most Important Features.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetal_sv_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetal_class_sv_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='PuOr').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
else:
    st.info(':information_source: *Please upload data to proceed*')