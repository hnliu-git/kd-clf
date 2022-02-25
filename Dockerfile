FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update \
    && apt-get install -y git locales && \
    locale-gen en_US.UTF-8

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD python3 finetune.py
