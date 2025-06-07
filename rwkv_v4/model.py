import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MAX_LENGTH, SOS_TOKEN
from rwkv_encoder import RWKV_TimeMix


# ====================================================================
# Encoder
# ===================================================================

class EncoderRNN(nn.Module):
    # The size of the input vocabulary, i.e., the number of unique words or tokens.
    # The size of the hidden state vector, which is the internal state of the RNN that gets updated over time as the model processes the sequence.
    
    # hidden_size = 128
    # encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)

    def __init__(self, input_vocabulary_size, hidden_size, dropout_p=0.1):
        # super().__init__() This is the simpler and more modern syntax introduced in Python 3. It works in most cases and is more concise. It implicitly refers to the immediate parent class of the current class (in this case, nn.Module).
        # This is the older, more explicit form that was commonly used in Python 2 and early versions of Python 3. In this syntax, you explicitly pass the current class (EncoderRNN) and the instance of the current class (self) as arguments to super().
        super(EncoderRNN, self).__init__()


        self.hidden_size = hidden_size

        #  The input_vocabulary_size represents the size of the input vocabulary, and hidden_size represents the size of each embedding vector (the dimensionality of the word representation).
        self.embedding = nn.Embedding(input_vocabulary_size, hidden_size)
        
        # First hidden_size (input_vocabulary_size): This refers to the dimensionality of the input features at each time step in the sequence. The GRU expects input vectors at each time step to have this size. In your case, the input to the GRU is coming from an embedding layer, which has already transformed word indices into dense vectors of size hidden_size.
        # Second hidden_size (hidden_size): This represents the dimensionality of the GRU's internal hidden state. The hidden state is what the GRU updates and maintains across time steps as it processes the sequence. The hidden state vector at each time step captures the "memory" of what the GRU has seen so far in the sequence. The hidden state is also the output of the GRU at each time step
        # batch_first=True: This tells the GRU that the first dimension of the input tensor corresponds to the batch size (i.e., the input should have shape [batch_size, sequence_length, hidden_size]).
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        
        # During training, dropout randomly sets a fraction of the output units (neurons) in a neural network to zero at each forward pass.
        # This is specified by a parameter p, which in your case is 0.1, meaning 10% of the neurons will be randomly dropped out.
        # This forces the remaining units to learn a more robust and generalized representation of the data.
        # During the test phase (or inference), dropout is disabled, and all neurons are active.
        # With dropout, the network can't rely on any specific neurons, as they might be dropped during any iteration. As a result, the model learns to distribute the information across many neurons, reducing overfitting and improving generalization to new data.
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # The input (which is a sequence of token indices) is first passed through the embedding layer, converting the indices into dense vectors of size hidden_size. Then, dropout is applied to the embedded vectors to randomly zero out some dimensions for regularization during training.
        embedded = self.dropout(self.embedding(input))
        # output: The output from the GRU for each time step (i.e., for each word in the sequence). It has a shape of [batch_size, sequence_length, hidden_size].
        # hidden: The hidden state after the final time step of the GRU for the last word in the sequence. This hidden state can be used to initialize the hidden state of a decoder in a sequence-to-sequence model.
        output, hidden = self.gru(embedded)
        return output, hidden
    

# ====================================================================
# RWKV Encoder
# ===================================================================

class RWKVEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, ctx_len=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx_len = ctx_len

        # Optional embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # A config object (like GPTConfig) with minimal fields
        class Config:
            def __init__(self, n_embd, n_layer, ctx_len, model_type, vocab_size):
                self.n_embd = n_embd
                self.n_layer = n_layer
                self.ctx_len = ctx_len
                self.model_type = model_type
                self.vocab_size = vocab_size

        config = Config(
            n_embd=hidden_size,
            n_layer=2,         # At least 2 layers to avoid ZeroDivisionError
            ctx_len=ctx_len,
            model_type='RWKV', # Optional descriptor
            vocab_size=vocab_size
        )

        # Create your RWKV_TimeMix layer
        self.rwkv_layer = RWKV_TimeMix(config, layer_id=0)

    def forward(self, input_tokens, xx_init=None, aa_init=None, bb_init=None, pp_init=None):
        """
        input_tokens: LongTensor [batch_size, seq_len]
        Returns: rwkv, e1, e2
        """
        # 1) Embed
        x = self.embedding(input_tokens)  # => [B, T, hidden_size]

        # 2) Forward pass through RWKV_TimeMix
        rwkv, e1, e2 = self.rwkv_layer(
            x, xx_init=xx_init, aa_init=aa_init, bb_init=bb_init, pp_init=pp_init
        )

        # You can compute hx, cx from e1 and e2 as necessary. Here we use e1 and e2 directly.
        hx = (e1 + e2) / 2  # You can modify this as needed
        cx = (e1 + e2) / 2  # Likewise, modify it to represent the cell state

        # Return rwkv as well it's used in the decoder to determine batch size.
        return rwkv, hx


# ====================================================================
# Decoder
# ===================================================================

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocabulary_size, device):
        super(DecoderRNN, self).__init__()
        # Maps each token in the target vocabulary (of size output_size) to a dense vector of size hidden_size.
        self.embedding = nn.Embedding(output_vocabulary_size, hidden_size)
        # Input size: hidden_size (embedding vector size).
        # Hidden size: hidden_size (dimensionality of GRU's internal state).
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Maps the GRU's hidden state to a vector of size output_size (corresponding to the vocabulary).
        # Not present in the encoder
        self.out = nn.Linear(hidden_size, output_vocabulary_size)
        self.device = device

    # encoder_outputs: The outputs of the encoder RNN (not used here but may be helpful in attention mechanisms).
    # encoder_hidden: The final hidden state of the encoder RNN (used to initialize the decoder's hidden state).
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # Extracts the batch size from encoder_outputs. Each batch contains multiple sequences.
        batch_size = encoder_outputs.size(0)
        # A tensor filled with the start-of-sequence (SOS) token, which initializes the decoding process.
        # Shape: [batch_size, 1] (one token per sequence).
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
        # Sets the initial hidden state of the decoder to the final hidden state of the encoder. This links the encoding and decoding processes.
        decoder_hidden = encoder_hidden
        # A list to store the outputs (predictions) for each time step.
        decoder_outputs = []

        # Iterates over the maximum sequence length to decode each token one by one.
        for i in range(MAX_LENGTH):
            # Calls the forward_step method to produce the next token and update the hidden state.
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            # Appends the decoder output to the list of outputs.
            decoder_outputs.append(decoder_output)

            # If the target sequence (target_tensor) is provided, use the actual token from the target as the next input to the decoder.
            # During training
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                # target_tensor[:, i]: Selects the i-th token for each sequence in the batch.
                # Unsqueeze(1): Adds a dimension to match the shape [batch_size, 1].
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                #  If no target is provided, use the decoder's own predictions as the next input.
                # Without teacher forcing: use its own predictions as the next input

                # decoder_output.topk(1): Selects the index of the highest probability token (predicted token).
                # this is not decoder outputs list!!!!!!!

                # decoder_output is the raw output from the decoder for the current time step. It has shape [batch_size, 1, output_size].
                # output_size is the size of the target vocabulary, and each element represents the unnormalized logit for each token in the vocabulary.
                # .topk(1) retrieves the highest value (the most likely token) along the last dimension (output_size).

                # topk returns two tensors:
                    # Values: The probability scores (not used here).
                    # Indices (topi): The token indices corresponding to the highest probability scores.
                _, topi = decoder_output.topk(1)

                # The topi tensor (indices of the predicted tokens) has shape [batch_size, 1].
                # squeeze(-1) removes the singleton dimension (1) at the last axis, resulting in a tensor of shape [batch_size].
                # Example:
                # Before squeezing: [[2], [4], [0]]
                # After squeezing: [2, 4, 0]

                # .detach():

                # Purpose: Detaches the tensor from the computation graph.
                # Normally, PyTorch keeps track of all operations on tensors to compute gradients during backpropagation. Here, we don't want gradients to flow through the prediction (because it's being used as an input for the next step, not part of the loss computation).
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
        
        # Combines the outputs for all time steps into a single tensor of shape [batch_size, MAX_LENGTH, output_size].
        # this is the list!!
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # Converts raw scores into log-probabilities over the vocabulary for each token. This is useful for loss calculation.
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # Outputs:
        # decoder_outputs: Log-probabilities over the vocabulary for the entire sequence.
        # decoder_hidden: The final hidden state of the decoder.
        # None: Placeholder for compatibility with training loops that expect additional outputs (e.g., attention weights).        
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        # Converts the input token index into its dense embedding vector.
        output = self.embedding(input)
        # Applies a ReLU activation to introduce non-linearity.
        output = F.relu(output)
        #Passes the embedding through the GRU, updating the hidden state.
        output, hidden = self.gru(output, hidden)
        # Maps the GRU output to a vector of size output_size (raw scores over the vocabulary).
        output = self.out(output)
        return output, hidden



# ====================================================================
# Attention Decoder
# ===================================================================

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights