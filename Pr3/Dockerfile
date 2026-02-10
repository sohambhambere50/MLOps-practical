# Use python official base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy all project files
COPY . .

# Run training when container starts
CMD ["python", "train.py"]
