import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

split_size = 80
parameter_n_estimators = 100
parameter_max_features = "auto"
parameter_min_samples_split = 10
parameter_min_samples_leaf = 10

parameter_random_state = 42
parameter_criterion = "mse"
parameter_bootstrap =  True
parameter_oob_score = False
parameter_n_jobs = 1


def build_model(df):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                               random_state=parameter_random_state,
                               max_features=parameter_max_features,
                               criterion=parameter_criterion,
                               min_samples_split=parameter_min_samples_split,
                               min_samples_leaf=parameter_min_samples_leaf,
                               bootstrap=parameter_bootstrap,
                               oob_score=parameter_oob_score,
                               n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    joblib.dump(rf, 'model.pkl')

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

def app():
    st.write("""Start off with uploading your CSV to train your model.""")

    with st.sidebar.header('Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

    with st.sidebar.header('Training Parameters'):
        st.sidebar.markdown("Split Size: " + str(split_size))

    st.subheader('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            #diabetes = load_diabetes()
            #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            #Y = pd.Series(diabetes.target, name='response')
            #df = pd.concat( [X,Y], axis=1 )

            #st.markdown('The Diabetes dataset is used as the example.')
            # st.write(df.head(5))

            # Boston housing dataset
            boston = load_boston()
            X = pd.DataFrame(boston.data, columns=boston.feature_names)
            Y = pd.Series(boston.target, name='response')
            df = pd.concat([X, Y], axis=1)

            st.markdown('The Boston housing dataset is used as the example.')
            st.write(df.head(5))

            build_model(df)