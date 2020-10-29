from train import initialize_for_train, train
from utils import check_files
from test import initialize_for_test, get_prediction
from params import Params

# initialize parameters
params = Params()
# check the existence of train / test data and create necessary folder
check_files(params)

# collect all inputs for training
data_loader, vocab_size, encoder, decoder, optimizer = initialize_for_train(params)
# training
train(params, data_loader, vocab_size, encoder, decoder, optimizer)

# collect all inputs for testing
data_loader, encoder, decoder = initialize_for_test(params)
# testing
get_prediction(data_loader, encoder, decoder, params)

