import argparse
import torch

import data_dif as data
from trainer_trans_unet import TrainerMultiple
from utils import *
import pickle
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument("fingerprint_dir", type=str,
                        help="Directory containing fingerprint and train values")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing real and fake images within 0_real and 1_fake subdirectories")
    parser.add_argument("--epoch", type=int, default=0, required=False,
                        help="Check point epoch to load")
    parser.add_argument("--batch", type=int, default=64, required=False,
                        help="Batch size")

    parsed_args = parser.parse_args()
    return parsed_args


def test_dif_directory(args: argparse.Namespace) -> (float, float):
    '''

    :param args: parser arguments (image directory, fingerprint directory, checkpoint epoch)
    :return: Accuracies for real and fake images
    '''

    model_ep = args.epoch
    images_dir = Path(args.image_dir)
    check_dir = Path(args.fingerprint_dir)

    check_existence(check_dir, False)
    check_existence(images_dir, False)

    with open(check_dir / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"

    hyper_pars['Device'] = torch.device(device_name)
    hyper_pars['Batch Size'] = args.batch

    print(f'Working on {images_dir.stem}')

    real_path_list = [list((images_dir / "0_real").glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    real_path_list = [ele for ele in real_path_list if ele != []][0]

    fake_path_list = [list((images_dir / "1_fake").glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    fake_path_list = [ele for ele in fake_path_list if ele != []][0]

    test_set = data.PRNUData(real_path_list, fake_path_list, hyper_pars, demand_equal=False,
                             train_mode=False)

    trainer = TrainerMultiple(hyper_pars, False)
    #trainer.load_stats(check_dir / f"chk_{model_ep}.pth")

    loss, acc = trainer.validate(test_set.get_loader())

    return loss, acc


if __name__ == '__main__':
    loss, acc = test_dif_directory(parse_arguments())
    print(f'Accuracy. {100 * acc:.1f}% | Loss {loss:.1f} ')
