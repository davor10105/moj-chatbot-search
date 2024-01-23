FROM python:3.11-bookworm

WORKDIR /search

RUN apt-get update && apt-get install libgl1 -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./search .

CMD ["python3", "run.py"]
