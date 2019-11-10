# Using official python runtime base image
FROM tiangolo/uwsgi-nginx-flask:python3.7

# Set the application directory
WORKDIR /app

# Install our requirements.txt
ADD requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip
RUN pip install -r requirements.txt

# Copy our code from the current folder to /app inside the container
ADD . /app
RUN python /app/training/predict-late-payers-basic-model.py

# Make port 80 available for links and/or publish
#EXPOSE 80

# Label
LABEL maintainer="Roy <findroy@outlook.in>"

# Make publicly accessible folder
VOLUME ["/app/public"]