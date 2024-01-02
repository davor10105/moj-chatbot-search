FROM python:3.11-bookworm

WORKDIR /search

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./search .

CMD ["python3", "run.py"]