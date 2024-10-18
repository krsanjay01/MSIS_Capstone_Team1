import argparse
import torch

import data_dif as data
from trainer_trans_unet import TrainerMultiple
from utils import *
import pickle
from pathlib import Path
from train_trans_unet import CustomDataset, load_data, create_dataloaders, normalize, val_transforms, load_full_test_data


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

    # Load hyperparameters
    with open(check_dir / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)

    # Set device
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"

    hyper_pars['Device'] = torch.device(device_name)
    hyper_pars['Batch Size'] = args.batch

    # Load data using the function defined in the training script
    print(f'Loading test data from {images_dir.stem}...')
    test_loader = load_full_test_data(images_dir, args.batch,val_transforms)

    print(f'Loaded {len(test_loader.dataset)} images for testing.')

    # Initialize the trainer
    trainer = TrainerMultiple(hyper_pars, train=False)

    # Load model weights
    checkpoint_path = check_dir / f"chk_{model_ep}.pth"
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=hyper_pars['Device']))
    print(f"Loaded model checkpoint from {checkpoint_path}")

    # Validate the model
    print('Validating...')
    loss, acc = trainer.validate(test_loader)

    return loss, acc


if __name__ == '__main__':
    loss, acc = test_dif_directory(parse_arguments())
    print(f'Accuracy. {100 * acc:.1f}% | Loss {loss:.1f} ')
