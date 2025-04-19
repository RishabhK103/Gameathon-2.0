
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies for Selenium and Brave Browser
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    apt-transport-https \
    ca-certificates \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set up Brave Browser repository
RUN curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list

# Update package lists and install Brave Browser
RUN apt-get update && apt-get install -y brave-browser \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir \
    -r requirements.txt



# Create necessary directories
RUN mkdir -p data/recent_averages

# Command to run when container starts
CMD ["python", "main.py"]