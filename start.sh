#!/bin/bash
# start.sh
# Install necessary dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Start the Streamlit application
echo "Starting the Streamlit application..."
streamlit run app.py
