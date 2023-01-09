import argparse
import yaml

from visualizer import *
from model import *
from trainer import *

import torch.backends.cudnn as cudnn
from torchsummary import summary


param_path = './param.yaml'
with open(param_path) as f:
    param = yaml.safe_load(f)


parser = argparse.ArgumentParser()

# path dir
parser.add_argument('--dataset-dir', default=param['dataset_dir'], type=str, help='dataset dir')
parser.add_argument('--test-dir', default=param['test_dir'], type=str, help='evaluation dataset dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='preprocess data dir')
parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model dir')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result dir')
parser.add_argument('--result-file', default=param['result_file'], type=str, help='result file name')
parser.add_argument('--machines', default=param['machines'], nargs='+', type=str, help='allowed processing machine')

parser.add_argument('--seed', default=param['seed'], type=int, help='random seed')
# model dir
parser.add_argument('--training-mode', default=param['training_mode'], type=str)
parser.add_argument('--version', default=param['version'], type=str, help='version')
# spectrogram features
parser.add_argument('--sr', default=param['sr'], type=int, help='STFT sampling rate')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT n_fft')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT win length')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT hop length')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT n_mels')
parser.add_argument('--frames', default=param['frames'], type=int, help='STFT time frames')
parser.add_argument('--power', default=param['power'], type=float, help='STFT power')
# training
parser.add_argument('--batch-size', default=param['batch_size'], type=int, help='batch size')
parser.add_argument('--epochs', default=param['epochs'], type=int, help='training epochs')
parser.add_argument('--early-stop', default=param['early_stop'], type=int, help='number of epochs for early stopping')
parser.add_argument('--lr', default=param['lr'], type=float, help='initial learning rate')
parser.add_argument('--num-workers', default=param['num_workers'], type=int, help='number of workers for dataloader')
parser.add_argument('--device-ids', default=param['device_ids'], nargs='+', type=int, help='gpu ids')
# model parameters
parser.add_argument('--channel-mul', default=param['channel_mul'], type=int, help='number of channel multiply')
parser.add_argument('--n-blocks', default=param['n_blocks'], type=int, help='number of residual blocks')
parser.add_argument('--n-groups', default=param['n_groups'], type=int, help='number of groups in conv layer')
parser.add_argument('--kernel-size', default=param['kernel_size'], type=int, help='conv kernel size')
# data augmentation
parser.add_argument('--aug-orig', default=param['aug_orig'], type=int, help='append original data')
parser.add_argument('--aug-mixup', default=param['aug_mixup'], type=int, help='append mixup data')
parser.add_argument('--aug-seg', default=param['aug_seg'], type=int, help='append seg data')


def set_random_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def preprocess():
    dirs = utils.select_dirs(args)
    for path in dirs:
        dataset.generate_train_dataset(args, path)
        dataset.generate_eval_dataset(args, path)
        dataset.generate_test_dataset(args, path)


def train_wavenet():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list = [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        model = WaveNet(n_blocks=args.n_blocks,
                        n_channel=args.n_mels,
                        n_mul=args.channel_mul,
                        frames=args.frames,
                        kernel_size=args.kernel_size,
                        n_groups=args.n_groups)

        training_dataloader, val_dataloader = dataset.load_wavenet_dataloader(args, machine_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            wn_trainer = WaveNetTrainer(args=args,
                                        machine_type=machine_type,
                                        visualizer=visualizer,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler)

            mean, inv_cov = wn_trainer.train(training_dataloader, val_dataloader)
            mean_list.append(mean)
            inv_cov_list.append(inv_cov)
    return mean_list, inv_cov_list


def test_wavenet(mean_list, inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov = mean_list[idx], inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'

        model = WaveNet(n_blocks=args.n_blocks,
                        n_channel=args.n_mels,
                        n_mul=args.channel_mul,
                        frames=args.frames,
                        kernel_size=args.kernel_size,
                        n_groups=args.n_groups)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            wn_trainer = WaveNetTrainer(args=args,
                                        machine_type=machine_type,
                                        visualizer=None,
                                        model=model,
                                        optimizer=None,
                                        scheduler=None)
            wn_trainer.test(mean, inv_cov)
    return


def train_resnet():
    dirs = utils.select_dirs(args)
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7
        model = ResNet(n_class=n_class)

        training_dataloader, val_dataloader = dataset.load_resnet_dataloader(args, machine_type)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            rn_trainer = ResNetTrainer(args=args,
                                       machine_type=machine_type,
                                       visualizer=visualizer,
                                       model=model,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       n_class=n_class)
            rn_trainer.train(training_dataloader, val_dataloader)


def test_resnet():
    for idx, machine_type in enumerate(args.machines):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'

        n_class = 6 if machine_type == 'ToyConveyor' else 7
        model = ResNet(n_class=n_class)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            rn_trainer = ResNetTrainer(args=args,
                                       machine_type=machine_type,
                                       visualizer=None,
                                       model=model,
                                       optimizer=None,
                                       scheduler=None,
                                       n_class=n_class)
            rn_trainer.test()


def train_mtl_class():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLSegVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLClass(n_blocks=args.n_blocks,
                         n_channel=args.n_mels,
                         n_mul=args.channel_mul,
                         frames=args.frames,
                         kernel_size=args.kernel_size,
                         n_groups=args.n_groups,
                         n_class=n_class)

        training_dataloader, val_dataloader = dataset.load_mtl_class_dataset(args, target_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))


        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            mtl_trainer = MTLClassTrainer(args=args,
                                          machine_type=machine_type,
                                          visualizer=visualizer,
                                          model=model,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          n_class=n_class)

        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_dataloader, val_dataloader)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list


def test_mtl_class(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLClass(n_blocks=args.n_blocks,
                         n_channel=args.n_mels,
                         n_mul=args.channel_mul,
                         frames=args.frames,
                         kernel_size=args.kernel_size,
                         n_groups=args.n_groups,
                         n_class=n_class)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            mtl_trainer = MTLClassTrainer(args=args,
                                          machine_type=machine_type,
                                          visualizer=None,
                                          model=model,
                                          optimizer=None,
                                          scheduler=None,
                                          n_class=n_class)
            mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def train_mrwn(is_sum=False):
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = WaveNetVisualizer()
        visualizer.add_machine_type(machine_type)

        if is_sum:
            model = MultiResolutionSumWaveNet(n_blocks=args.n_blocks,
                                              n_channel=args.n_mels,
                                              n_mul=args.channel_mul,
                                              frames=args.frames,
                                              kernel_size=args.kernel_size,
                                              n_groups=args.n_groups)
        else:
            model = MultiResolutionWaveNet(n_blocks=args.n_blocks,
                                           n_channel=args.n_mels,
                                           n_mul=args.channel_mul,
                                           frames=args.frames,
                                           kernel_size=args.kernel_size,
                                           n_groups=args.n_groups)

        training_dataloader, val_dataloader = dataset.load_wavenet_dataloader(args, machine_type)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))

        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            mrwn_trainer = MultiResolutionWaveNetTrainer(args=args,
                                                         machine_type=machine_type,
                                                         visualizer=visualizer,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler)

            mean, inv_cov, block_mean, block_inv_cov = mrwn_trainer.train(training_dataloader, val_dataloader)
            mean_list.append(mean)
            inv_cov_list.append(inv_cov)
            block_mean_list.append(block_mean)
            block_inv_cov_list.append(block_inv_cov)

    return mean_list, inv_cov_list, block_mean_list, block_inv_cov_list


def test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list, is_sum=False):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, block_mean, block_inv_cov = mean_list[idx], inv_cov_list[idx], block_mean_list[idx], block_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'

        if is_sum:
            model = MultiResolutionSumWaveNet(n_blocks=args.n_blocks,
                                              n_channel=args.n_mels,
                                              n_mul=args.channel_mul,
                                              frames=args.frames,
                                              kernel_size=args.kernel_size,
                                              n_groups=args.n_groups)
        else:
            model = MultiResolutionWaveNet(n_blocks=args.n_blocks,
                                           n_channel=args.n_mels,
                                           n_mul=args.channel_mul,
                                           frames=args.frames,
                                           kernel_size=args.kernel_size,
                                           n_groups=args.n_groups)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            mrwn_trainer = MultiResolutionWaveNetTrainer(args=args,
                                                         machine_type=machine_type,
                                                         visualizer=None,
                                                         model=model,
                                                         optimizer=None,
                                                         scheduler=None)
            mrwn_trainer.test(mean, inv_cov, block_mean, block_inv_cov)


def train_mtl_seg():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLClassVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLSeg(n_blocks=args.n_blocks,
                       n_channel=args.n_mels,
                       n_mul=args.channel_mul,
                       frames=args.frames,
                       kernel_size=args.kernel_size,
                       n_groups=args.n_groups,
                       n_class=n_class)

        training_dataloader, val_dataloader = dataset.load_mtl_class_dataset(args, target_dir, is_seg=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))
        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            mtl_trainer = MTLSegmentationTrainer(args=args,
                                                 machine_type=machine_type,
                                                 visualizer=visualizer,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 n_class=n_class)


        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_dataloader, val_dataloader)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list


def test_mtl_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLSeg(n_blocks=args.n_blocks,
                       n_channel=args.n_mels,
                       n_mul=args.channel_mul,
                       frames=args.frames,
                       kernel_size=args.kernel_size,
                       n_groups=args.n_groups,
                       n_class=n_class)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            mtl_trainer = MTLSegmentationTrainer(args=args,
                                                 machine_type=machine_type,
                                                 visualizer=None,
                                                 model=model,
                                                 optimizer=None,
                                                 scheduler=None,
                                                 n_class=n_class)
            mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def train_mtl_class_seg():
    dirs = utils.select_dirs(args)
    mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = [], [], [], []
    for idx, target_dir in enumerate(dirs):
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(dirs)}] {target_dir}')

        machine_type = os.path.split(target_dir)[1]

        visualizer = MTLClassSegVisualizer()
        visualizer.add_machine_type(machine_type)

        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLClassSeg(n_blocks=args.n_blocks,
                            n_channel=args.n_mels,
                            n_mul=args.channel_mul,
                            frames=args.frames,
                            kernel_size=args.kernel_size,
                            n_groups=args.n_groups,
                            n_class=n_class)

        training_dataloader, val_dataloader = dataset.load_mtl_class_seg_dataset(args, target_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_dataloader))
        with torch.cuda.device(args.device_ids):
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)

            mtl_trainer = MTLClassSegTrainer(args=args,
                                             machine_type=machine_type,
                                             visualizer=visualizer,
                                             model=model,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             n_class=n_class)

        mean, inv_cov, score_mean, score_inv_cov = mtl_trainer.train(training_dataloader, val_dataloader)

        mean_list.append(mean)
        inv_cov_list.append(inv_cov)
        score_mean_list.append(score_mean)
        score_inv_cov_list.append(score_inv_cov)
    return mean_list, inv_cov_list, score_mean_list, score_inv_cov_list


def test_mtl_class_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list):
    for idx, machine_type in enumerate(args.machines):
        mean, inv_cov, score_mean, score_inv_cov = mean_list[idx], inv_cov_list[idx], score_mean_list[idx], score_inv_cov_list[idx]
        print('\n' + '=' * 60)
        print(f'[{idx + 1}/{len(args.machines)}] {machine_type}')
        model_path = f'{args.model_dir}/{args.version}/{machine_type}/checkpoint_best_model.pth.tar'
        n_class = 6 if machine_type == 'ToyConveyor' else 7

        model = MTLClassSeg(n_blocks=args.n_blocks,
                            n_channel=args.n_mels,
                            n_mul=args.channel_mul,
                            frames=args.frames,
                            kernel_size=args.kernel_size,
                            n_groups=args.n_groups,
                            n_class=n_class)

        with torch.cuda.device(args.device_ids):
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            args.dp = False
            if len(args.device_ids) > 1:
                args.dp = True
                model = torch.nn.DataParallel(model, device_ids=args.device_ids)
            mtl_trainer = MTLClassSegTrainer(args=args,
                                             machine_type=machine_type,
                                             visualizer=None,
                                             model=model,
                                             optimizer=None,
                                             scheduler=None,
                                             n_class=n_class)
            mtl_trainer.test(mean, inv_cov, score_mean, score_inv_cov)


def main(args):
    preprocess()

    if args.training_mode == 'WaveNet':
        mean_list, inv_cov_list = train_wavenet()
        test_wavenet(mean_list, inv_cov_list)

    elif args.training_mode == 'ResNet':
        train_resnet()
        test_resnet()

    elif args.training_mode == 'MTL_class':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_class()
        test_mtl_class(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)

    elif args.training_mode == 'MTL_seg':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_seg()
        test_mtl_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)

    elif args.training_mode == 'MRWN':
        mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = train_mrwn()
        test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list)

    elif args.training_mode == 'MRSWN':
        mean_list, inv_cov_list, block_mean_list, block_inv_cov_list = train_mrwn(is_sum=True)
        test_mrwn(mean_list, inv_cov_list, block_mean_list, block_inv_cov_list, is_sum=True)

    elif args.training_mode == 'MTL_class_seg':
        mean_list, inv_cov_list, score_mean_list, score_inv_cov_list = train_mtl_class_seg()
        test_mtl_class_seg(mean_list, inv_cov_list, score_mean_list, score_inv_cov_list)



if __name__ == "__main__":
    args = parser.parse_args()
    set_random_everything(param['seed'])
    print(f'Model path: {args.version}')

    # gpu settings
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device = torch.device(f'cuda:{args.device_ids[0]}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    main(args)