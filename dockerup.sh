#!/bin/bash

docker build -t voxtral .

docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home:/home voxtral bash -c "cd /home/gabriel/Documents/projects/voxtral && bash"