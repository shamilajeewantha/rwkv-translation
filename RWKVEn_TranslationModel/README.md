#Instructions for RWKV-V4 training
    go to the RWKVEn_TranslationModel folder
    pip install torch numpy pandas tqdm
    pip install Ninja
    pip install deepspeed
    export RWKV_FLOAT_MODE=fp16
    rm -rf RWKV-LM
    git clone https://github.com/BlinkDL/RWKV-LM.git
    cd RWKV-LM/RWKV-v4
    change the precision in train.py file from bf16 to fp16
    change the data set path in train.py file accordingly
    downgrade pip to install lightning
    pip install --upgrade pip==23.0.1
    pip install pytorch-lightning==1.6.0
    run train.py

#Instructions for RWKV encoder+ LSTM decoder translator
export RWKV_FLOAT_MODE=fp16
    go to the RWKVEn_TranslationModel folder
    rwkv encoder is in rwkvEncoder.py file
    change the dataset path as required
    run translation.py

