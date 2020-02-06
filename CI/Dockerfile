FROM python:3.6
WORKDIR .

RUN apt-get update
RUN apt-get -y install swig
RUN pip install --upgrade pip
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision virtualenv
