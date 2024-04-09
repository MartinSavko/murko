FROM python:3.10-slim

RUN apt-get update --fix-missing && \
    apt-get install -y vim libsasl2-dev python-dev-is-python3 libldap2-dev libssl-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y vim && apt-get autoremove -y 

RUN mkdir /opt/app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py /opt/app

COPY model.h5 /opt/app

COPY murko.py /opt/app 

WORKDIR /opt/app

CMD ["python","server.py","-p","8008"]
