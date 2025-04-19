FROM python:3.11.4-slim

LABEL team="Sinister 6"
LABEL version="0.1.0"
LABEL description="Optimisitation algorithm algorithm to build a fantacy cricket team for IPL 2025, FIFS gamethon"
LABEL github="https://github.com/RishabhK103/Gameathon-2.0"

# System deps and Chrome setup
RUN apt-get update -qq && \
	apt-get install -y --no-install-recommends \
	wget \
	unzip \
	libasound2 \
	libatk-bridge2.0-0 \
	libnss3 \
	libx11-xcb1 \
	libxcomposite1 \
	libxdamage1 \
	libxrandr2 \
	libgbm1 \
	libgtk-3-0 \
	xdg-utils \
	ca-certificates \
	fonts-liberation \
	gnupg \
	libdrm2 && \
	rm -rf /var/lib/apt/lists/*

# Download and install Chrome (121.0.6167.85)
RUN wget -q -O chrome-linux64.zip https://bit.ly/chrome-linux64-121-0-6167-85 && \
	unzip chrome-linux64.zip && \
	rm chrome-linux64.zip && \
	mv chrome-linux64 /opt/chrome && \
	ln -s /opt/chrome/chrome /usr/local/bin/google-chrome

# Make a “brave” alias so your script’s brave_path works
RUN ln -s /usr/local/bin/google-chrome /usr/bin/brave

# Download and install ChromeDriver
RUN wget -q -O chromedriver-linux64.zip https://bit.ly/chromedriver-linux64-121-0-6167-85 && \
	unzip -j chromedriver-linux64.zip chromedriver-linux64/chromedriver && \
	rm chromedriver-linux64.zip && \
	mv chromedriver /usr/local/bin/chromedriver && \
	chmod +x /usr/local/bin/chromedriver && \
	ln -s /usr/local/bin/chromedriver /usr/bin/chromedriver

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
