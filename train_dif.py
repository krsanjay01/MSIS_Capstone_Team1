from tqdm import tqdm
import torch
import argparse
import data_dif as data
from trainer_dif import TrainerMultiple
from utils import *
from pathlib import Path
import pickle



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument("image_dir", type=str,
                        help="Directory containing real and fake images within 0_real and 1_fake subdirectories.")
    parser.add_argument("checkpoint_dir", type=str,
                        help="Directory to save checkpoints. Model is not saved, only fingerprint and statistics.")
    parser.add_argument("--e", type=int, default=100, required=False,
                        help="Amount of train iterations.")
    parser.add_argument("--f", type=int, default=5, required=False,
                        help="Check point frequency.")
    parser.add_argument("--lr", type=float, default=5e-4, required=False,
                        help="Learning rate")
    parser.add_argument("--tr", type=int, default=512, required=False,
                        help="Amount of train samples per real/fake class.")
    parser.add_argument("--cs", type=int, default=256, required=False,
                        help="Crop size (w=h)")
    parser.add_argument("--a", type=float, default=1.0, required=False,
                        help="Alpha - augmentations")
    parser.add_argument("--b", type=bool, default=False, required=False,
                        help="Booster loss")
    parser.add_argument("--bs", type=int, default=64, required=False,
                        help="Booster loss")
    
    parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')                   
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="swin_unet/configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parsed_args = parser.parse_args()
    if parsed_args.dataset == "Synapse":
        parsed_args.root_path = os.path.join(parsed_args.root_path, "train_npz")
    return parsed_args


def train_model(args: argparse.Namespace) -> None:

    data_root = Path(args.image_dir)
    check_dir = Path(args.checkpoint_dir)


    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"

    device = torch.device(device_name)

    hyper_pars = {'Epochs': args.e, 'Factor': args.f, 'Noise Type': 'uniform', "Train Size": args.tr,
                  'Noise STD': 0.03, 'Inp. Channel': 16, 'Batch Size': 64,
                  'LR': 5e-4, 'Device': device, 'Crop Size': (args.cs, args.cs), 'Margin': 0.01,
                  'Out. Channel': 3, 'Arch.': 32, 'Depth': 4, 'Alpha': args.a, 'Boost': args.b,
                  'Concat': [1, 1, 1, 1]}

    check_existence(check_dir, True)
    check_existence(data_root, False)

    print('Preparing Data Sets...')

    real_data_root = data_root / "0_real"
    fake_data_root = data_root / "1_fake"

    real_path_list = [list(real_data_root.glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    real_path_list = [ele for ele in real_path_list if ele != []][0]

    fake_path_list = [list(fake_data_root.glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    fake_path_list = [ele for ele in fake_path_list if ele != []][0]

    train_set = data.PRNUData(real_path_list, fake_path_list, hyper_pars, demand_equal=False,
                             train_mode=False)
    train_loader = train_set.get_loader()

    pickle.dump(hyper_pars, open((check_dir / 'train_hypers.pt'), 'wb'))

    print('Preparing Trainer...')
    trainer = TrainerMultiple(hyper_pars, args).to(hyper_pars['Device'])

    epochs = list(range(1, hyper_pars['Epochs'] + 1))
    pbar = tqdm(total=len(epochs), desc='')

    for ep in epochs:
        pbar.update()

        for residual, labels in train_loader:
            trainer.train_step(residual, labels)

        if (ep % hyper_pars['Factor']) == 0:
            if ep > 0:
                trainer.save_stats(check_dir / ('chk_' + str(ep) + '.pt'))

        pbar.postfix = f'Loss C {np.mean(trainer.train_loss[-10:]):.3f} ' + \
                       f'| Fake C {np.mean(trainer.train_corr_f[-10:]):.3f} | Real C {np.mean(trainer.train_corr_r[-10:]):.3f}'

    trainer.save_stats(check_dir / ('chk_' + str(hyper_pars['Epochs']) + '.pt'))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train_model(parse_arguments())
