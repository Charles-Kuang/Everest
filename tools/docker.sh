#! /bin/bash
#docker run -it --rm -v $PWD:/mnt/everest --gpus all zllai/everest:1.3 bash
docker run -it -v $PWD:/mnt/everest --gpus all docker_copy bash