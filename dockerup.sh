#!/bin/bash

docker build -t voxtral .
docker run -it --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nfsdata:/nfsdata voxtral bash -c "cd /nfsdata/gabrielc/voxtral && bash"