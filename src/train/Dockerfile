# Select a Python 3.9-slim official runtime as the base image
FROM python:3.9-slim

# Define the working directory within the container as /app
WORKDIR /app

# Copy the contents of the current directory to the container's /app directory
COPY . /app

# Install the required packages listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to execute when the container is launched, running the train.py script
CMD ["python", "src/train/train.py"]
