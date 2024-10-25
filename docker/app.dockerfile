# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project into the container
COPY . /code

ENV CPUINFO_NO_WARNINGS=1

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "pamod.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
