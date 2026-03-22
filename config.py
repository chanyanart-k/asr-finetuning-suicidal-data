import streamlit as st

def get_model_config() -> dict:
    """
        Load model fileIDs and local paths from Streamlit secrets. 
    """
    return {
        "FINETUNED_FILE_ID": st.secrets["gdrive"]["FINETUNED_FILE_ID"],
        "BASE_FILE_ID": st.secrets["gdrive"]["BASE_FILE_ID"],
        "FINETUNED_MODEL_PATH": st.secrets["gdrive"]["FINETUNED_MODEL_PATH"],
        "BASE_MODEL_PATH":  st.secrets["gdrive"]["BASE_MODEL_PATH"],
    }