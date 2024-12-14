# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port that your app will run on
EXPOSE 5000

# Step 6: Define environment variables (optional, for production settings)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Step 7: Run the application using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
