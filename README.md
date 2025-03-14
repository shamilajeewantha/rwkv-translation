# rwkv-translation

setup miniconda or whatever

for v4
conda create -n rwkv4 python=3.7 -y
conda activate rwkv4
conda install nvidia/label/cuda-11.3.1::cuda-toolkit -y
nvcc --version
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

conda install -c conda-forge cudatoolkit=11.4.0 -y
conda install -c conda-forge cudatoolkit-dev=11.4.0 -y

pip install torch==1.10.0+cu114torchvision==0.11.1+cu114 torchaudio==0.10.0+cu114 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install deepspeed==0.7.0
pip install pytorch-lightning==1.9.5
pip install ninja
pip install wandb

conda deactivate
conda env remove -n rwkv4

for v6
conda create -n rwkv6 python=3.12 -y
conda activate rwkv6
nvcc --version (should be the sytemwide one)
conda install nvidia/label/cuda-12.8.1::cuda-toolkit -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -y
pip install wandb pytorch-lightning==1.9.5 deepspeed
conda deactivate # If the environment is currently active
conda env remove -n rwkv6

python train.py
python infer.py

https://www.geeksforgeeks.org/how-to-visualize-training-progress-in-pytorch/
