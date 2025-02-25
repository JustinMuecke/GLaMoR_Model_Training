FROM python:3.10

WORKDIR /app
COPY requirements.txt /

RUN pip install torch
RUN pip install -r /requirements.txt

COPY . /app

CMD ["python3", "/app/test.py"]