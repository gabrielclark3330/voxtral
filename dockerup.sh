#!/bin/bash

docker build -t voxtral .

#'"device=0,1"'
docker run -it --gpus=all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nfsdata:/nfsdata -v /mnt:/mnt voxtral bash -c "cd /nfsdata/gabrielc/voxtral && bash"