# Use an official Python image with poetry installed
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy the application code
COPY . .

# Set environment variables
ENV TOKENIZERS_PARALLELISM=True \
    TF_CPP_MIN_LOG_LEVEL=3

# Expose ports if needed (e.g., for an API server)
# EXPOSE 8000

# Command to run the script
CMD ["poetry", "run", "python", "main.py", "--question", "How are you?"]
