# Use a lightweight Python base image
FROM python:3.10-slim

# Install tools to download the Solidity static binary
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      wget \
      unzip \
 && rm -rf /var/lib/apt/lists/*

# Download and install the Solidity 0.8.0 static binary
RUN wget -qO /usr/bin/solc \
      https://github.com/ethereum/solidity/releases/download/v0.8.0/solc-static-linux \
 && chmod +x /usr/bin/solc

# Set working directory
WORKDIR /app

# Copy requirements first (to leverage Docker layer caching)
COPY requirements.txt .  
RUN pip install -r requirements.txt

# Copy all application code
COPY . .

# Make the server entrypoint executable
RUN chmod +x server/entrypoint.sh

# Default command
CMD ["sh", "/app/server/entrypoint.sh"]
