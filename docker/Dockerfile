FROM python:3.8.0

ARG username=saitotatsuki 
ARG groupid=1000
ARG userid=1000

RUN apt-get update

#python
COPY ./docker/requirements.txt /
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

#login user
RUN groupadd -g $groupid $username \
    && useradd -m -r -u $userid -g $username $username

WORKDIR /Workdir
USER $username
