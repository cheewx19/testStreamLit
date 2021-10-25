
import streamlit as st
import trainModel
import recommendCourse
# Page layout

navigation = {
    "Train Your Model": trainModel,
    "Recommend Courses": recommendCourse
}

def main():
    st.title("""Machine Learning Recommendation System""")
    recommendCourse.app()

if __name__ == '__main__':
        main()