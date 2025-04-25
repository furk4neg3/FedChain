# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port (optional, e.g. if your server runs on 5000)
EXPOSE 5000

# Default command (optional, override with docker-compose or CLI)
CMD ["python", "server/server.py"]
