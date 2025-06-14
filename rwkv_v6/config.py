import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract values
MAX_LENGTH = config["MAX_LENGTH"]
LEARNING_RATE = config["LEARNING_RATE"]
HIDDEN_SIZE = config["HIDDEN_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
SOS_TOKEN = config["SOS_TOKEN"]  # Start of sentence token: 0
EOS_TOKEN = config["EOS_TOKEN"]  # End of sentence token: 1
DATA_PATH = config["DATA_PATH"]
EPOCHS = config["EPOCHS"]
RWKV_FLOAT_MODE = config.get("RWKV_FLOAT_MODE")  
HEAD_SIZE = config.get("HEAD_SIZE")  # Default value if not specified

# Print to confirm values are loaded correctly
if __name__ == "__main__":
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"HIDDEN_SIZE: {HIDDEN_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SOS_token: {SOS_TOKEN}")
    print(f"EOS_token: {EOS_TOKEN}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"RWKV_FLOAT_MODE: {RWKV_FLOAT_MODE}")
    print(f"HEAD_SIZE: {HEAD_SIZE}")
