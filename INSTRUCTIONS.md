# Pre-req: docker must be installed
#### Docker can be installed from https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04

## Install docker
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
## Check if it's running
sudo systemctl status docker

# Start model docker
docker build -t model .
docker run -itd -p 6000:6000 --rm --name model -v $PWD:/app model /bin/bash

# Start flask docker
docker build -t flaskapp .
docker run -it -p 5000:5000 -e FLASK_APP=app.py --rm --name flaskapp -e FLASK_DEBUG=1 -v $PWD:/app --volumes-from model flaskapp

# Verify if volumes work
docker container exec -it flaskapp sh
then navigate to respective folder.

# Start nginx docker
docker build -t nginx .
docker run -it --name nginx -p 80:80 nginx

# Use docker compose
docker-compose up --build