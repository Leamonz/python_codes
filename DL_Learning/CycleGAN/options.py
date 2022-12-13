import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='../../dataset/CycleGAN/monet2photo/train')
parser.add_argument('--sample_path', type=str, default='./samples')
parser.add_argument('--model_path', type=str, default='./models')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--image_channels', type=int, default=3)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lambda_cycle', type=float, default=10.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--sample_interval', type=int, default=200)
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--save_model', action='store_true')

parser.add_argument('--result_path', type=str, default='./results')

args = parser.parse_args()
print(args)
