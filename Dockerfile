# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and virtual environment
COPY requirements.txt /app/
COPY venv /app/venv



# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /app/

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"] 