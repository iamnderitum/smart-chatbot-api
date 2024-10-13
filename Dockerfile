# Step 1: Use official Python image from the Docker Hub as a Base Image
FROM python:3.9-slim

# # Step 2: Set the Working directory inside the container
WORKDIR /app

# # Step 3: Copy the requirements file into the container.
COPY requirements.txt .

# Step 4: Install the required Python Packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: # Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Download NLTK data
RUN python -m nltk.downloader punkt

# Step 7: Expose the port your Flask app runs on
EXPOSE 5000

# Step 8: Define the command to run the application
CMD ["python", "app.py"]
