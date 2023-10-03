import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=100,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-4,  help='learning rate')
parser.add_argument('--arch',       type=str, default="Unet",  help='models archtechture')
parser.add_argument('--batchsize',   type=int,   default=10,    help='training batch size')
parser.add_argument('--clip_size',   type=int,   default=5,    help='a clip size')
parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=60,    help='every n epochs decay learning rate')
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
parser.add_argument('--sgd',      default=False,  action='store_true')

parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

parser.add_argument('--data_root',      type=str, default='/home/zl/Workspace/video_object_seg/GSFM/data/breast_lesion_seg',           help='the training rgb images root')
parser.add_argument('--split_file',      type=str, default='train_list',           help='the training rgb images root')
parser.add_argument('--save_path',           type=str, default='./Checkpoint/SPNet/',    help='the path to save models and logs')


opt = parser.parse_args()

