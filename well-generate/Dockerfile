FROM python:3.8-slim

WORKDIR /app

RUN pip install numpy pandas scikit-learn nltk torch

RUN python -m nltk.downloader punkt

COPY . /app

EXPOSE 80

CMD ["python", "train.py"]