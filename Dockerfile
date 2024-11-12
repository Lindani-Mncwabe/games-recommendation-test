# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app/contacts-chat-recommendations
WORKDIR /app/contacts-chat-recommendations

# Copy the requirements from the root of the project
COPY requirements.txt /app/

# Install system dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the contents of the project (root and subfolders) into the container
COPY . /app/

# Set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS="ayoba-credentials.json"

# Expose port
EXPOSE 8000

# Run the app.py in the `contacts-chat-recommendations` folder
CMD ["python", "app.py"]
