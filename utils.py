import os
import shutil


def check_files(params):
    assert os.path.exists(params.image_train)
    assert os.path.exists(params.annotation_train)
    assert os.path.exists(params.image_test)
    if not os.patzh.exists(params.encoder_save):
        os.makedirs(params.encoder_save)
    if not os.path.exists(params.decoder_save):
        os.makedirs(params.decoder_save)
    if os.path.exists(params.test_save):
        shutil.rmtree(params.test_save)
        print("All files under ", params.test_save, " has been deleted. ")
    os.makedirs(params.test_save)


