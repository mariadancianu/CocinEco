import os
import socket

import streamlit as st

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to CocinEco! 👋")


def is_running_on_streamlit_cloud():
    # Check if "is_cloud" exists in secrets and is set to true
    running_on_cloud = st.secrets.get("general", {}).get("is_cloud", False)

    if running_on_cloud:
        pass
    else:
        st.warning("running on local")
    return running_on_cloud


def init_cocineco():
    if is_running_on_streamlit_cloud():
        st.session_state.use_authentication = True
        st.session_state.running_in_cloud = True
    else:
        st.session_state.use_authentication = False
        st.session_state.running_in_cloud = False

    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = False


def authenticate_user():
    correct_username = "cocineco-admin"
    correct_password = "cocineco-admin-2025"

    # Prompt the user for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    st.session_state.authenticated_user = (
        username == correct_username
    )  # and password == correct_password)
    st.markdown("Please enter your Username and Password")
    if st.button("Login"):
        if st.session_state.authenticated_user:
            st.success("Login successful!")
        else:
            st.error(
                "Please login to access CocinEcoBot. To obtain a Username and Password please contact: contact.a3isp@gmail.com."
            )


def main():
    init_cocineco()
    if not st.session_state.authenticated_user and st.session_state.use_authentication:

        authenticate_user()
    else:
        st.session_state.authenticated_user = True
        st.sidebar.success("Select a Planner Above.")

        st.success("Login successful!")


main()
