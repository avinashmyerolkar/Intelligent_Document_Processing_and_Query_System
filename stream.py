import streamlit as st
import os
import requests 
# Set the title of the app
st.title("Intelligent Document Processing and Query System")

# Display a welcome message
#st.write("Hello")

# Create a sidebar for file upload
uploaded_files = st.sidebar.file_uploader("Upload 10 Docs", type=['pdf'], accept_multiple_files=True)

# Define the directory where files will be saved
UPLOAD_DIR = '/home/avinash_m_yerolkar/frontend/download-dir'

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Save uploaded files to the specified directory
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
    st.sidebar.write(f"Uploaded {len(uploaded_files)} files successfully.")

# Create a text input field for chatbot queries
st.header("Chatbot Query")
user_query = st.text_input("Ask your question:")

# Process and display the response from the chatbot
if user_query:
    # Send the user query to the Flask API
    response = requests.post('http://164.52.196.197:7539/chat', json={'query': user_query})
    
    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        st.write(response_data['response'])
    else:
        st.write("Error: Unable to get a response from the chatbot.")
