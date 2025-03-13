FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY backend/ ./backend/
COPY data_for_train/ ./data_for_train/
COPY results/ ./results/
COPY ollama_0220.py ./backend/

# Set environment variables
ENV A2I2_BASE_DIR=/app
ENV PYTHONPATH=/app

# Expose the port
EXPOSE $PORT

# Start the application
CMD cd backend && uvicorn server:app --host 0.0.0.0 --port $PORT 