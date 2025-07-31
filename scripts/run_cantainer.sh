#/bin/bash

docker run -it -v "$(pwd)/..:/home/$(whoami)/Epona" --gpus all epona