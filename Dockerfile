
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Copy Datadog configuration files
COPY datadog-agent/conf.d/python.d/conf.yaml /etc/datadog-agent/conf.d/python.d/conf.yaml

# Create the logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8080

# Run app.py when the container launches using ddtrace-run
CMD ["ddtrace-run", "python", "app.py"]
