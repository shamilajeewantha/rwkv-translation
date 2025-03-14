from translation import *

print("evaluate")
import os
print("Current working directory:", os.getcwd())






# Load saved models
#please change the path to the saved weights
# encoder.load_state_dict(torch.load('/home/thulasi/UPEKSHA/RWKV/RWKV-LM/RWKV-v4/encoder1.pth'))
# decoder.load_state_dict(torch.load('/home/thulasi/UPEKSHA/RWKV/RWKV-LM/RWKV-v4/decoder1.pth'))

encoder.load_state_dict(torch.load('../training_weights/eng-fra/encoder3.pth'))
decoder.load_state_dict(torch.load('../training_weights/eng-fra/decoder3.pth'))

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Evaluate
evaluateRandomly(encoder, decoder)


#######please comment out training and saving weights in the training module before running this module