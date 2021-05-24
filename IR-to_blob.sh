src_vol='/home/auro/azure-percept-advanced-development/machine-learning-notebooks/train-from-scratch/outputs'

docker run --rm -v /media/auro/RAID\ 5/sushi-u-net-segment/outputs:/working \
       -w /working openvino/ubuntu18_dev:2021.1 /bin/bash compile.sh

# docker run -it -v $src_vol:/working -w /working openvino/ubuntu18_dev:2021.3 /bin/bash 
