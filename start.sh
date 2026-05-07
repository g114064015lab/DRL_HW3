#!/bin/bash
# ==============================================================================
# start.sh
# Purpose: Initialize project structure, set environment variables, install 
#          dependencies, and launch the training application in the background.
# ==============================================================================

echo "========================================="
echo " Starting DRL_HW3 Environment Setup"
echo "========================================="

# 1. Create necessary directories for professional organization
echo "[1/4] Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p results
mkdir -p docs

# 2. Export environment variables (Hyperparameters)
echo "[2/4] Setting environment variables..."
export DEFAULT_LEARNING_RATE="0.001"
export DEFAULT_BATCH_SIZE="32"
export DEFAULT_EPOCHS="1000"
export DEFAULT_EPSILON_DECAY="0.99"
echo "  -> LEARNING_RATE: $DEFAULT_LEARNING_RATE"
echo "  -> BATCH_SIZE: $DEFAULT_BATCH_SIZE"

# 3. Install dependencies
echo "[3/4] Checking and installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "  -> WARNING: requirements.txt not found!"
fi

# 4. Launch the application in the background
echo "[4/4] Launching Streamlit Application..."
nohup streamlit run app.py > logs/streamlit_app.log 2>&1 &
APP_PID=$!
echo $APP_PID > logs/app.pid

echo "========================================="
echo " Application is running in the background."
echo " Process ID: $APP_PID"
echo " Log output is redirected to logs/streamlit_app.log"
echo " Use ./ending.sh to stop the application and clean up."
echo "========================================="
