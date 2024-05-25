FROM docker://tensorflow/tensorflow:2.15.0-gpu

RUN mkdir /opt/app

RUN pip install --upgrade pip && pip install numpy scipy zmq psutil scikit-image matplotlib simplejpeg seaborn

COPY train.py /opt/app

COPY dataset_loader.py /opt/app

COPY generate_masks.py /opt/app

COPY model.h5 /opt/app

COPY murko.py /opt/app

WORKDIR /opt/app

CMD ["python","server.py","-p","8008"]
