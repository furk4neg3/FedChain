FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Ensure our script is executable
RUN chmod +x server/entrypoint.sh

# Default (will be overridden by Compose entrypoint)
CMD ["sh", "/app/server/entrypoint.sh"]
