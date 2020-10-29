from data_loader import get_loader
import numpy as np
import cv2
import os
import torch
from model import Encoder, Decoder

def initialize_for_test(params):
    data_loader = get_loader(params, mode='test')
    encoder_file = os.path.join(params.encoder_save, 'epoch-%d.pkl' % params.num_epochs)
    decoder_file = os.path.join(params.decoder_save, 'epoch-%d.pkl' % params.num_epochs)
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = Encoder(params)
    decoder = Decoder(params, vocab_size)
    encoder.eval()
    decoder.eval()

    # Load the trained weights.
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    encoder.to(params.device)
    decoder.to(params.device)
    return data_loader, encoder, decoder


def generate_sentence(ids, data_loader):
    # <start> or <end>. the number (0, 1) is fixed because in function build_vocab of class Vocabulary
    # it always add start_word first, end_word second, and unk_word third.
    omit = [0, 1]
    words = []
    for i in ids:
        if i  not in omit:
            word = data_loader.dataset.vocab.idx2word[i]
            words.append(word)
            if word == '.':     # end of sentence
                break
    sentence = ' '.join(words)
    return sentence


def get_prediction(data_loader, encoder, decoder, params):
    captions_collection = open(params.test_captions, 'w')

    for i in range(params.num_test_cases):
        orig_image, image = next(iter(data_loader))
        # use squeeze because it's possible that original has 1st dimension as its id in a batch
        # orig_image is torch.tensor, so need to transfer to np
        orig_image = cv2.cvtColor(np.squeeze(orig_image.numpy()), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(params.test_save, "fig" + str(i) + ".png"), orig_image)
        image = image.to(params.device)
        features = encoder(image).unsqueeze(1)  # the first dimension should be id in batch
        ids = decoder.predict(features)
        sentence = generate_sentence(ids, data_loader)
        captions_collection.write("fig" + str(i) + ' -> ' + sentence + '\n')
        if (i+1) % params.pred_print_every == 0:
            print("Predict: ", i+1, " / ", params.num_test_cases)
    captions_collection.close()



