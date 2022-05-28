import argparse

from SBIR.Baseline_FGSBIR.Code.dataset import get_dataloader

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

        parser.add_argument('--dataset_name', type=str, default='ShoeV2')
        parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
        parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                            help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
        parser.add_argument('--root_dir', type=str, default='./../')
        parser.add_argument('--batchsize', type=int, default=16)
        parser.add_argument('--nThreads', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        # parser.add_argument('--max_epoch', type=int, default=200)
        parser.add_argument('--max_epoch', type=int, default=100)
        parser.add_argument('--eval_freq_iter', type=int, default=100)
        parser.add_argument('--print_freq_iter', type=int, default=1)

        hp = parser.parse_args()
        dataloader_Train, dataloader_Test = get_dataloader(hp)
