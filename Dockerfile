FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY input ./input
COPY output ./output

ENTRYPOINT ["python", "app/extractor.py"]