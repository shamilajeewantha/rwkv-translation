setup miniconda or whatever

for v4
conda create -n rwkv4 python=3.7 -y
conda activate rwkv4
conda install -c conda-forge cudatoolkit=11.3.1 -y
conda install -c conda-forge cudatoolkit-dev=11.4.0 -y

conda run pip install deepspeed==0.7.0
conda run pip install pytorch-lightning==1.9.5
conda run pip install ninja
conda run pip install wandb

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
