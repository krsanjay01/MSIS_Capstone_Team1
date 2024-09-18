import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pickle

# Import the modified TrainerMultiple class
from trainer_trans_unet import TrainerMultiple

# Data transforms
# Define normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Training data transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Validation data transformations
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Ensure 3 channels
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("image_dir", type=str,
                        help="Directory containing real and fake images within 0_real and 1_fake subdirectories.")
    parser.add_argument("checkpoint_dir", type=str,
                        help="Directory to save checkpoints and trained model.")
    parser.add_argument("--epochs", type=int, default=20, required=False,
                        help="Number of training epochs.")
    parser.add_argument("--save_freq", type=int, default=5, required=False,
                        help="Checkpoint save frequency.")
    parser.add_argument("--lr", type=float, default=1e-4, required=False,
                        help="Learning rate.")
    parser.add_argument("--train_size", type=int, default=512, required=False,
                        help="Number of training samples per class.")
    parser.add_argument("--crop_size", type=int, default=224, required=False,
                        help="Crop size (width and height).")
    parser.add_argument("--alpha", type=float, default=1.0, required=False,
                        help="Alpha value for data augmentation.")
    parser.add_argument("--batch_size", type=int, default=64, required=False,
                        help="Batch size.")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to use for training.")

    parsed_args = parser.parse_args()
    return parsed_args


def load_data(image_dir):
    image_dir = Path(image_dir)

    real_dir = image_dir / '0_real'
    fake_dir = image_dir / '1_fake'

    # Collect image paths
    real_images = []
    fake_images = []

    for ext in ['jpg', 'jpeg', 'png']:
        real_images.extend(list(real_dir.glob(f'*.{ext}')))
        fake_images.extend(list(fake_dir.glob(f'*.{ext}')))

    real_labels = [0] * len(real_images)
    fake_labels = [1] * len(fake_images)

    image_paths = real_images + fake_images
    labels = real_labels + fake_labels

    return image_paths, labels


def create_dataloaders(image_paths, labels, batch_size, train_transforms, val_transforms, train_size,
                       validation_split=0.2):
    # Limit the number of training samples per class
    indices_per_class = {0: [], 1: []}
    for idx, label in enumerate(labels):
        indices_per_class[label].append(idx)

    # Shuffle indices
    np.random.shuffle(indices_per_class[0])
    np.random.shuffle(indices_per_class[1])

    # Select specified number of samples per class
    selected_indices = indices_per_class[0][:train_size] + indices_per_class[1][:train_size]

    # Split into training and validation sets
    np.random.shuffle(selected_indices)
    split = int(np.floor(validation_split * len(selected_indices)))
    train_indices = selected_indices[split:]
    val_indices = selected_indices[:split]

    # Prepare datasets
    train_image_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_image_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transforms)
    val_dataset = CustomDataset(val_image_paths, val_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_model(args: argparse.Namespace) -> None:
    data_root = Path(args.image_dir)
    check_dir = Path(args.checkpoint_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hyper_pars = {
        'Epochs': args.epochs,
        'Save_Freq': args.save_freq,
        'LR': args.lr,
        'Batch Size': args.batch_size,
        'Device': device,
        'Inp. Channel': 3,  # RGB images
        'Out. Channel': 2,  # Binary classification
    }

    check_dir.mkdir(parents=True, exist_ok=True)

    print('Preparing Data Sets...')

    # Load data
    image_paths, labels = load_data(data_root)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        image_paths, labels, args.batch_size, train_transforms, val_transforms, args.train_size
    )

    # Save hyperparameters
    pickle.dump(hyper_pars, open((check_dir / 'train_hypers.pt'), 'wb'))

    print('Preparing Trainer...')
    trainer = TrainerMultiple(hyper_pars)

    best_val_accuracy = 0.0

    for epoch in range(1, hyper_pars['Epochs'] + 1):
        print(f"Epoch {epoch}/{hyper_pars['Epochs']}")

        # Training phase
        trainer.train_loss = []
        trainer.train_accuracy = []
        for images, labels in tqdm(train_loader, desc='Training'):
            trainer.train_step(images, labels)

        avg_train_loss = np.mean(trainer.train_loss)
        avg_train_accuracy = np.mean(trainer.train_accuracy)

        # Validation phase
        val_loss, val_accuracy = trainer.validate(val_loader)

        # Learning rate scheduler step
        trainer.scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save model and stats at specified frequency
        if (epoch % hyper_pars['Save_Freq']) == 0 or epoch == hyper_pars['Epochs']:
            model_save_path = check_dir / f'chk_{epoch}.pth'
            torch.save(trainer.model.state_dict(), model_save_path)
            print(f"Checkpoint saved at {model_save_path}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = check_dir / 'best_model.pth'
            torch.save(trainer.model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")

    torch.cuda.empty_cache()

#Usage
#python train_dif.py /path/to/image_dir /path/to/checkpoint_dir --epochs 20 --batch_size 64 --lr 1e-4 --save_freq 5

if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)

