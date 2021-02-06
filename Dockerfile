FROM python:3.7-buster
RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y \
   curl \
   python3-setuptools && \
apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -f get-pip.py
RUN pip3 install --no-cache numpy Pillow seldon-core

RUN pip install -r requirements.txt
RUN pip install -r external-private-packages.txt


# Seldon Core specific
COPY . /microservice
WORKDIR /microservice

ENV MODEL_NAME BrandGenderPredictor
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE

EXPOSE 5000