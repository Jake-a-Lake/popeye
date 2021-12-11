FROM ubuntu:20.04 as base
#FROM czentye/opencv-video-minimal as base
LABEL maintainer "Steven Farmer"
ENV OPEN_CV_VERSION=4.3.0
ENV MAKE_THREADS=12
ENV APP_DIR=/app


RUN mkdir -p ${APP_DIR} && \
    chmod 777 ${APP_DIR}
RUN apt-get update && \
    apt-get install -y curl apt-transport-https 
RUN apt-get install -y gnupg gnupg2 gnupg1
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - 
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | tee /etc/apt/sources.list.d/msprod.list
RUN apt-get update 
#&& apt-get install -y mssql-tools unixodbc-dev

#RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list 
RUN ACCEPT_EULA=Y apt-get install -y unixodbc-dev odbcinst1debian2
RUN ACCEPT_EULA=Y apt-get install -y mssql-tools
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends wget git ffmpeg build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
    libatlas-base-dev gfortran python3-dev python3-pip ca-certificates   && \
    pip3 install numpy && \
    apt-get clean autoclean && \
    apt-get autoremove --yes

#ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools libodbc1 unixodbc-dev unixodbc libpq-dev
RUN ls -lsa
#RUN pip3 install --no-cache-dir -r .requirements.txt
#RUN pip3 install -r requirements2.txt
#Set the Work Dir Now
WORKDIR /app

#Add Ze Python SOurce
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#Run the last few pip3 statement.  Some of these could be moved up, but this works for now.
# COPY ./requirements.txt ./requirements.txt
# RUN pip3 install --no-cache-dir -r ./requirements.txt
COPY . .
# apt-get clean autoclean && \
# apt-get autoremove --yes && \
# rm -rf /var/lib/{apt,dpkg,cache,log}/ 

# #Install dlib with python bindings - do we need this?
# RUN apt install git && \
#     git clone https://github.com/davisking/dlib.git && \
#     cd dlib && \
#     mkdir build ; cd build; cmake ..; cmake --build . && \
#     cd .. && \
#     python3 setup.py install
#Leaving this here for Kubernetes/OpenShift
#USER 1001





# ##--------------SET UP THE DEBUGGER---------------
# FROM base as debug
# # Debug image reusing the base
# # Install dev dependencies for debugging
# RUN pip install debugpy
# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE 1
# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED 1

# FROM base as prod
# # Production image
# RUN pip install gunicorn
# ##------------END DEBUGGER STUFF-------------------

#Add all the Python source
#ADD . / . /
#COPY . .
#ADD . .

#Run the darn thing already.
#EXPOSE 5010
#CMD ["python3", "countcars.py"]
#CMD gunicorn mlapi:app

## for use with debugger config
# CMD ["gunicorn", "--reload", "--bind", "0.0.0.0:5010", "mlapi:app"]

# I think we need to add this to the container start up command instead 
# because this just ended up in an image build
#RUN python3 mlapi.py