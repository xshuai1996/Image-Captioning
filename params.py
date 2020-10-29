from torchvision import transforms
import os
import torch
import torch.nn as nn

class Params(object):
    def __init__(self):
        """ Wrapper class for various parameters. """
        self.vocab_threshold = 7  # Set the minimum word count threshold
        self.transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))])
        self.transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # WHAT IS THIS????????????
                                 (0.229, 0.224, 0.225))])
        self.batch_size_train = 64  # batch size in training
        self.batch_size_test = 1  # batch size in testing
        self.vocabulary_filename = 'vocab.pkl'
        self.start_word = "<start>"  # start_word: Special word denoting sentence start
        self.end_word = "<end>"  # end_word: Special word denoting sentence end
        self.unk_word = "<unk>"  # unk_word: Special word denoting unknown words
        self.image_train = os.path.join("cocoapi", "images", "train2014")  # path of a folder
        self.annotation_train = os.path.join("cocoapi", "annotations", "captions_train2014.json")
        self.image_test = os.path.join("cocoapi", "images", "test2014")  # path of a folder
        self.annotation_test = os.path.join("cocoapi", "annotations", "image_info_test2014.json")
        self.embed_size = 512  # dimensionality of image and word embeddings
        self.hidden_size = 512  # number of features in hidden state of the RNN decoder
        self.num_epochs = 5  # number of training epochs
        self.save_every = 1  # determines frequency of saving model weights
        self.print_every = 100  # determines window for printing average loss
        self.log_file = 'training_log.txt'  # name of file with saved training loss and perplexity
        # Move models to GPU if CUDA is available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LSTM_layers = 1
        # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects
        # num_layers greater than 1
        self.LSTM_dropout = 0.4 if self.LSTM_layers > 1 else 0
        self.maxlen_generate = 20   # max length of generated captions
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        self.encoder_save = os.path.join("models", "encoder")
        self.decoder_save = os.path.join("models", "decoder")
        self.test_save = "test cases"
        self.test_captions = os.path.join(self.test_save, "captions_collections.txt")
        self.num_test_cases = 50        # the number of images will be used in test
        self.inference_mode = 'sampling'      # one of 'beamsearch' or 'sampling'
        self.num_candidates = 20        # number of candidates for beamsearch mode
        self.pred_print_every = 1      # print the progress of prediction every x iterations
