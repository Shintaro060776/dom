FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py .
COPY processed.csv .
COPY processed_after_data.csv .
COPY train.py .

CMD ["python", "./train.py"]
