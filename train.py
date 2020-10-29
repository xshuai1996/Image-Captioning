from model import Encoder, Decoder
import torch.utils.data as data
import numpy as np
import os
import torch
from data_loader import get_loader
import math


def initialize_for_train(params):
    """ initialize before training. """
    data_loader = get_loader(params, mode='train')
    vocab_size = len(data_loader.dataset.vocab)     # the total number of keys in the word2idx dictionary
    print('Total number of tokens in vocabulary:', vocab_size)

    encoder = Encoder(params).to(params.device)
    decoder = Decoder(params, vocab_size).to(params.device)

    # Specify the learnable parameters of the model.
    learnable_params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(learnable_params)
    return data_loader, vocab_size, encoder, decoder, optimizer

def train(params, data_loader, vocab_size, encoder, decoder, optimizer):
    # Set the total number of training steps per epoch
    # math.ceil return the smallest integer value which is greater
    # than or equal to the specified expression or an individual number
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    # prepare to record average loss
    f = open(params.log_file, 'w')

    for epoch in range(1, params.num_epochs + 1):
        for i_step in range(1, total_step + 1):
            # the lengths of captions in training set vary greatly from around 5 to around 55.
            # in every step we want to randomly choose a length, then filter out all captions have this length,
            # then randomly select the number of captions among them.
            # the following function randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # get the data batch with the given indices
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(params.device)
            captions = captions.to(params.device)

            # Zero the gradients.
            encoder.zero_grad()
            decoder.zero_grad()

            # Pass the inputs through the encoder-decoder model.
            features = encoder(images)
            outputs = decoder(features, captions)

            loss = params.criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, params.num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

            f.write(stats + '\n')
            f.flush()

            if i_step % params.print_every == 0:
                print(stats)

        # Save the weights.
        if epoch % params.save_every == 0:
            torch.save(encoder.state_dict(), os.path.join(params.encoder_save, 'epoch-%d.pkl' % epoch))
            torch.save(decoder.state_dict(), os.path.join(params.decoder_save, 'epoch-%d.pkl' % epoch))
    f.close()

























# COCO API: https://github.com/cocodataset/cocoapi