#!/bin/bash
# ==============================================================================
# ending.sh
# Purpose: Gracefully shutdown the application, organize outputs, compress 
#          results into an archive, and clean up temporary files.
# ==============================================================================

echo "========================================="
echo " Stopping DRL_HW3 Environment"
echo "========================================="

# 1. Stop background processes
echo "[1/4] Stopping background processes..."
if [ -f "logs/app.pid" ]; then
    APP_PID=$(cat logs/app.pid)
    echo "  -> Killing Process ID: $APP_PID"
    kill $APP_PID 2>/dev/null
    rm logs/app.pid
    echo "  -> Process gracefully terminated."
else
    echo "  -> No PID file found. Attempting pkill fallback..."
    pkill -f "streamlit run app.py"
fi

# 2. Organize and prepare outputs
echo "[2/4] Organizing outputs..."
# Mocking a results dump (in a real scenario, app.py would write to results/)
touch results/training_summary.txt
echo "Training completed successfully." > results/training_summary.txt

# 3. Compress results
echo "[3/4] Compressing logs and results..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="run_archive_${TIMESTAMP}.tar.gz"
tar -czf $ARCHIVE_NAME logs/ results/
echo "  -> Archive created: $ARCHIVE_NAME"

# 4. Clean up temporary files
echo "[4/4] Cleaning up temporary files and caches..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf logs/*
rm -rf results/*

echo "========================================="
echo " Shutdown complete! All clean."
echo " Results have been backed up to $ARCHIVE_NAME"
echo "========================================="
