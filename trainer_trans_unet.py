import matplotlib.pyplot as plt
import numpy as np

from torch import optim, nn
import torch
from tqdm import tqdm

from dcnn_loader import load_denoiser
#import model_trans
from utils import calc_even_size, produce_spectrum

from torch.optim import lr_scheduler

import torch.nn.functional as F
from trans_unet.vit_seg_modeling_classifier import VisionTransformer
from trans_unet.vit_seg_configs_classifier import get_b16_config
from trans_unet.vit_seg_configs_classifier import get_r50_b16_config

relu = nn.ReLU()


class TrainerMultiple(nn.Module):
    def __init__(self, hyperparams, train=True):
        super(TrainerMultiple, self).__init__()

        # Hyperparameters
        self.init_lr = hyperparams['LR']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.batch_size = hyperparams['Batch Size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model configuration
        self.config = get_r50_b16_config()
        self.config.img_size = (224, 224)
        self.config.in_channels = self.ch_i
        self.config.n_classes = 2  # Binary classification
        self.config.num_classes = self.config.n_classes
        self.config.use_cls_token = True

        # Initialize VisionTransformer
        self.model = VisionTransformer(
            self.config,
            img_size=self.config.img_size,
            num_classes=self.config.num_classes,
            zero_head=True
        ).to(self.device)
        self.model.load_from(weights=np.load(self.config.pretrained_path))

        # Optimizer setup with different learning rates
        encoder_params = list(self.model.transformer.parameters())
        classifier_params = list(self.model.classification_head.parameters())

        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': self.init_lr * 0.1},
            {'params': classifier_params, 'lr': self.init_lr}
        ], weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics
        self.train_loss = []
        self.train_accuracy = []

        # Validation metrics
        self.val_loss = []
        self.val_accuracy = []

    def train_step(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device).long()

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(images)  # Outputs shape: [batch_size, num_classes]

        # Compute loss
        loss = self.criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        # Record loss
        self.train_loss.append(loss.item())

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels).item()
        accuracy = correct / labels.size(0)
        self.train_accuracy.append(accuracy)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.val_loss.append(avg_loss)
        self.val_accuracy.append(accuracy)
        return avg_loss, accuracy

    def save_stats(self, path):
        self.calc_centers()

        data_dict = {'Fingerprint': self.fingerprint,
                     'Train Real': self.train_corr_r,
                     'Train Fake': self.train_corr_f,
                     'Loss': self.train_loss}

        torch.save(data_dict, path)

    def load_stats(self, path):
        if self.device.type == 'cpu':
            data_dict = torch.load(path, map_location=torch.device('cpu'))
        elif self.device.type == 'mps':
            data_dict = torch.load(path, map_location=torch.device('mps'))
        else:
            data_dict = torch.load(path)

        self.train_loss = data_dict['Loss']
        self.train_corr_r = data_dict['Train Real']
        self.train_corr_f = data_dict['Train Fake']
        self.fingerprint = data_dict['Fingerprint']