FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files (excluding venv)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on (change if needed)
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
