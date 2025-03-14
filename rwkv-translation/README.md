setup miniconda or whatever

conda create -n rwkv6 python=3.12 -y
conda activate rwkv6
nvcc --version (should be the sytemwide one)
conda install nvidia/label/cuda-12.8.1::cuda-toolkit -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -y
pip install wandb pytorch-lightning==1.9.5 deepspeed -y
conda deactivate # If the environment is currently active
conda env remove -n rwkv6

python train.py
python infer.py

https://www.geeksforgeeks.org/how-to-visualize-training-progress-in-pytorch/
