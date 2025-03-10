from tqdm import tqdm
import time
import torch
import torch.optim as optim
from utils import showPlot
import torch.nn as nn
from config import HIDDEN_SIZE, BATCH_SIZE, LEARNING_RATE
from model import EncoderRNN, AttnDecoderRNN
from data_preprocessor import get_dataloader

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

# During training:

# Forward Pass: Compute predictions and loss.
# Backward Pass (.backward()):
    # Computes the gradient ∇L for each parameter.
    # These gradients are stored in the .grad attribute of each parameter.

# Optimizer Step (.step()):
#     Uses the gradients to update parameters:
#     Where η is the learning rate.



# In PyTorch, every trainable parameter (weights, biases) has an associated .grad attribute.
# python
# for param in model.parameters():
#     print(param.grad)  # Prints the gradient for each parameter

# Before calling .backward(), param.grad is None.
# After .backward(), param.grad holds the computed gradients.



# Why Do We Flatten?
# Let’s say we have the following setup:

# Batch size = 2 (we process two sequences at once).
# Sequence length = 3 (each sequence contains 3 time steps).
# Vocabulary size = 4 (our model predicts words from a vocabulary of 4 words).

# Shape: (batch_size=2, sequence_length=3, vocab_size=4)
# decoder_outputs = torch.tensor([
#     [  # First sequence (batch element 1)
#         [2.0, 1.0, 0.1, -1.0],  # Prediction at time step 1
#         [1.2, 2.5, -0.5, 0.0],  # Prediction at time step 2
#         [0.3, -1.0, 2.1, 0.5]   # Prediction at time step 3
#     ],
#     [  # Second sequence (batch element 2)
#         [1.5, 2.3, -0.2, 0.3],  # Prediction at time step 1
#         [2.1, -0.8, 1.0, 1.4],  # Prediction at time step 2
#         [-0.5, 0.8, 1.9, 2.2]   # Prediction at time step 3
#     ]
# ])

# Shape: (batch_size=2, sequence_length=3)
# target_tensor = torch.tensor([
#     [0, 1, 2],  # Correct words for the first sequence
#     [3, 2, 1]   # Correct words for the second sequence
# ])


# After flattening:
# decoder_outputs.view(-1, decoder_outputs.size(-1))  # Becomes (6, 4)
# target_tensor.view(-1)  # Becomes (6,)

# decoder_outputs = torch.tensor([
#     [2.0, 1.0, 0.1, -1.0],  # Time step 1, seq 1
#     [1.2, 2.5, -0.5, 0.0],  # Time step 2, seq 1
#     [0.3, -1.0, 2.1, 0.5],  # Time step 3, seq 1
#     [1.5, 2.3, -0.2, 0.3],  # Time step 1, seq 2
#     [2.1, -0.8, 1.0, 1.4],  # Time step 2, seq 2
#     [-0.5, 0.8, 1.9, 2.2]   # Time step 3, seq 2
# ])

# target_tensor = torch.tensor([0, 1, 2, 3, 2, 1])

# If we directly pass (2, 3, 4) and (2, 3), PyTorch will throw an error because CrossEntropyLoss expects:
# A 2D logits matrix (num_samples, num_classes), not (batch, seq, vocab).
# A 1D target tensor (num_samples,), not (batch, seq).

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Decoder return decoder_outputs, decoder_hidden, attentions
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=LEARNING_RATE):

    plot_losses = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Progress", unit="epoch"):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # print('Loss: %.4f' % loss)
        plot_losses.append(loss)
        showPlot(plot_losses)
            

# ====================================================================
# Main Training
# ====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


input_lang, output_lang, train_dataloader = get_dataloader(BATCH_SIZE, device)

encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, device).to(device)

print(encoder)
print(decoder)

train(train_dataloader, encoder, decoder, 100)
print("Training completed successfully.")

# Define file paths
# TODO: Change the paths as needed
encoder_path = "encoder.pth"
decoder_path = "decoder.pth"

# Save state dictionaries
torch.save(encoder.state_dict(), encoder_path)
torch.save(decoder.state_dict(), decoder_path)
print(f"Models saved successfully: {encoder_path}, {decoder_path}")