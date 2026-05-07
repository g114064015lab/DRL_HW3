#!/bin/bash
# ending.sh
echo "Stopping the Streamlit application..."

# Attempt to gracefully kill the Streamlit process
pkill -f "streamlit run app.py"

# Clean up any Python cache directories
echo "Cleaning up python cache..."
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Application stopped and cleanup complete."
