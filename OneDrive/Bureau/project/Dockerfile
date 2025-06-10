# Use an official Python image as base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files from your current directory to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app on container start
CMD ["streamlit", "run", "sepsis_App.py", "--server.port=8501", "--server.address=0.0.0.0"]
