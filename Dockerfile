FROM python:3.8-slim-buster
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR ./

COPY . .

CMD ["sh", "run_project.sh"]
