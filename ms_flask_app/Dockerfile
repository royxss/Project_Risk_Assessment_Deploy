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

# Make port 5000 available for links and/or publish
EXPOSE 5000

# Label
LABEL maintainer="Roy <findroy@outlook.in>"

# Define our command to be run when launching the container
CMD flask run --host=0.0.0.0 --port=5000
#CMD ["gunicorn", "app:app", "-b", "0.0.0.0:80", "--log-file", "-", "--access-logfile", "-", "--workers", "4", "--keep-alive", "0"]