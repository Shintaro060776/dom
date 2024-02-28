FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio

COPY train.py /app/

COPY cleaned_dataset.csv /app/

CMD ["python", "train.py"]