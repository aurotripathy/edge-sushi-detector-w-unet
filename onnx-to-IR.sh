# Volume to bind to the OpenVINO container
# pwd = !pwd
# src_vol = os.path.join(pwd[0], 'outputs')
# src_vol

# In [ ]:

# Convert in the OpenVINO container
# 
# src_vol='/home/auro/azure-percept-advanced-development/machine-learning-notebooks/train-from-scratch/outputs'
# src_vol='/media/auro/RAID\ 5/sushi-u-net-segment/outputs'


sudo docker run -it -v /media/auro/RAID\ 5/sushi-u-net-segment/outputs:/working -w /working openvino/ubuntu18_dev:2021.1 \
         python3 "/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py" \
         --input_model "./onnx/bananas.onnx" -o "./intel" --input "input" --output "output" --scale 255


