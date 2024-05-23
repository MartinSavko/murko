FROM tensorflow/tensorflow:2.15.0-gpu

RUN mkdir /opt/app

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[server]

COPY server.py /opt/app

COPY model.h5 /opt/app

COPY murko.py /opt/app 

WORKDIR /opt/app

CMD ["python","server.py","-p","8008"]
    