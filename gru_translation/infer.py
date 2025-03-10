from model import EncoderRNN, AttnDecoderRNN
from config import HIDDEN_SIZE, EOS_TOKEN
import torch
import random
from data_preprocessor import prepareData, tensorFromSentence

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, n=10):
    correct_count = 0  # Initialize the counter for correct translations

    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words).replace('<EOS>', '').strip()
        print('<', output_sentence)

        # Check if the model's output matches the target translation
        if output_sentence == pair[1]:
            print("Correct")
            correct_count += 1
        else:
            print("Incorrect")

        print('')  # Empty line for readability

    # Calculate the percentage of correct translations
    accuracy = (correct_count / n) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{n})")


# ====================================================================
# Main Inference
# ====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

# Initialize the models again before loading weights
encoder_loaded = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
decoder_loaded = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, device).to(device)

# Define file paths
encoder_path = "encoder.pth"
decoder_path = "decoder.pth"

# Load saved state dictionaries
encoder_loaded.load_state_dict(torch.load(encoder_path))
decoder_loaded.load_state_dict(torch.load(decoder_path))

print("Models loaded successfully!")

encoder_loaded.eval()
decoder_loaded.eval()
evaluateRandomly(encoder_loaded, decoder_loaded, 200)