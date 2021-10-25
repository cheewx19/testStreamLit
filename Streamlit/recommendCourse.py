import streamlit as st
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
import joblib
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from streamlit import caching

def app():
    caching.clear_cache()
    global tms
    global course_details
    global course_categories

    with st.sidebar.header('Upload TMS data'):
        tms_file = st.sidebar.file_uploader(
            "Upload your file", type=["csv"], key=1)
        if tms_file is not None:
            tms = pd.read_csv(tms_file)
        else:
            tms = None

    with st.sidebar.header('Upload Course Details data'):
        course_file = st.sidebar.file_uploader(
            "Upload your file", type=["xlsx"], key=2)
        if course_file is not None:
            course_details = pd.read_excel(course_file, sheet_name=1)
            course_categories = pd.read_excel(course_file, sheet_name='WBS')
        else:
            course_details,course_categories = None, None

    # if (st.sidebar.button('Press to use Example Dataset')):
    #     boston = load_boston()
    #     X = pd.DataFrame(boston.data, columns=boston.feature_names)
    #     Y = pd.Series(boston.target, name='response')
    #     df_view = pd.concat([X, Y], axis=1)
        
    #     st.markdown('The Boston housing dataset is used as the example.')
    #     st.write(df_view.head(5))

    if tms is not None and course_details is not None and course_categories is not None:
        global cust_name
        st.markdown('Please enter a name to feed into the recommender system.')
        cust_name = st.text_input("Customer Name:")
        if cust_name:   
            preprocessing()
            createMatrix()
            runModel()
    else:
        st.write("Please upload the relevant datasets on the side to start.")

def preprocessing():

    st.subheader('1. Data Pre-Processing')
    st.markdown('- Mapping subcategories to Title and Course Code')
    st.markdown('- Mapping Course Code to Title')
    course_and_category = course_details[['Code', 'Title', 'SubCategory']]

    title_to_subcategory = course_details[['Title', 'SubCategory']]
    title_to_subcategory = title_to_subcategory.set_index('Title').to_dict()['SubCategory']

    code_to_subcategory = course_details[['Code', 'SubCategory']]
    code_to_subcategory = code_to_subcategory.set_index('Code').to_dict()['SubCategory']

    tms['SubCategory'] = np.nan
    tms['SubCategory'] = tms['Course Reference Number'].apply(lambda x: title_to_subcategory[x] if x in title_to_subcategory \
                                                          else (code_to_subcategory[x] if x in code_to_subcategory else np.nan))

    title_to_code = course_details[['Title', 'Code']]
    title_to_code = title_to_code.set_index('Title').to_dict()['Code']
    
    tms['CourseCode'] = np.nan
    tms['CourseCode'] = tms['Course Reference Number'].apply(lambda x: title_to_code[x] if x in title_to_code \
                                                          else x)
    st.write(tms.head())

def createMatrix():
    st.subheader('2. Creation of Matrix')
    global matrix
    placeholder = st.empty()
    placeholder.write("Loading Matrix...")

    tms['Name (As in NRIC)'] = tms['Name (As in NRIC)'].str.upper()
    
    matrix = tms[['Name (As in NRIC)', 'CourseCode']]   

    y = pd.get_dummies(matrix.CourseCode)

    global final_matrix
    final_matrix = pd.concat([matrix['Name (As in NRIC)'], y], axis=1)
    final_matrix = final_matrix.groupby('Name (As in NRIC)')[final_matrix.columns[1:].tolist()].sum()
    final_matrix.index = final_matrix.index.rename('Name') 
    placeholder.empty()
    placeholder.markdown('The below matrix will be used for the recommender model.')
    st.write(final_matrix.head())

    if matrix['Name (As in NRIC)'].nunique() == final_matrix.index.nunique():
        st.write("Names are grouped correctly")
    else:
        st.write("Names are grouped incorrectly")
    
def runModel():
    st.subheader("3. Recommend Courses")
    st.write("**User-based:** For a user U, with a set of similar users determined based on rating vectors consisting of given item ratings, the rating for an item I, which hasn’t been rated, is found by picking out N users from the similarity list who have rated the item I and calculating the rating based on these N ratings.")
    st.write("**Item-based:** For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, the rating by a user U, who hasn’t rated it, is found by picking out N items from the similarity list that have been rated by U and calculating the rating based on these N ratings.")
    
    placeholder = st.empty()
    placeholder.write("Preparing the Model...")

    new_matrix = deepcopy(matrix)
    new_matrix['rating'] = 1
    new_matrix = new_matrix.groupby(['Name (As in NRIC)','CourseCode']).agg('sum').reset_index()

    reader = Reader(rating_scale=(1,5))
    sim_options = {
    "name": "cosine",
    "user_based": True,
    }
    algo = KNNBasic(k=5, sim_options=sim_options)

    count = 0
    increment = 5000
    while count < new_matrix.shape[0]:
        new_matrix_subset = new_matrix.iloc[count:count+increment,:]
        data = Dataset.load_from_df(new_matrix_subset, reader)
        trainingSet = data.build_full_trainset()
        algo.fit(trainingSet)
        count += increment
    placeholder.empty()

    n = 5
    df = pd.DataFrame(columns =['course', 'score'])  
    for course in list(final_matrix.columns):
        new_row = {'course': course, 'score': algo.predict(cust_name, course).est}
        df = df.append(new_row, ignore_index=True)
    df.sort_values(by='score', ascending=False, inplace=True)
    st.markdown('These are the recommended courses for ' + cust_name)
    st.write(df.iloc[:n])

    
