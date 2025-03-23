FROM python:3.11

LABEL team="Sinister 6"
LABEL version="0.1.0"
LABEL description="Optimisitation algorithm algorithm to build a fantacy cricket team for IPL 2025, FIFS gamethon"
LABEL github="https://github.com/RishabhK103/Gameathon-2.0"

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py", "--build", "--update", "1"]
