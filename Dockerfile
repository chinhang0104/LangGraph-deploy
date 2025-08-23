FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# Copy only requirements first to leverage caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app ./app
COPY ./README.md ./
COPY ./package[s] ./packages

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/code/app

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8001"]