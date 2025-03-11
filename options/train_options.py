from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualization parameters
        parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=20, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--unet_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--arch_lr', type=float, default=1e-4, help='initial learning rate for architecture')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # training parameters for dcgan
        parser.add_argument('--lr_dcgan', type=float, default=4e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_d_dcgan', type=float, default=4e-4, help='initial learning rate for adam')
        parser.add_argument('--model_1', type=str, default='DCGAN', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
        parser.add_argument('--download', type=str, default='False')
        parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
        parser.add_argument('--cuda',  type=bool, default=True, help='Availability of cuda')
        parser.add_argument('--cuda_index', type=int, default=1, help='The number of epochs to run')
        parser.add_argument('--channels', type=int, default=1, help='The number of epochs to run')

        parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
        parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
        parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')
        
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
        parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
        parser.add_argument('--is_train', type=str, default='True')

        # training parameter for unet
        parser.add_argument('--unet_learning_rate', type=float, default=1e-5, help='Learning rate')
        parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
        parser.add_argument('--test_percent', type=float, default=29.2, help='Percent of the data that is used as validation (0-100)')
        parser.add_argument('--val_percent', type=float, default=5.0, help='Percent of the data that is used as validation (0-100)')
        parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
        parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        parser.add_argument('--classes', type=int, default=1, help='Number of classes')
        parser.add_argument('--loss_lambda', type=float, default=1.0, help='Learning rate')
        parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")
        parser.add_argument('--model_dir', type=str, default='./checkpoint', help='saves results here.')
        parser.add_argument('--aug_type', type=str, default='rotate', help='saves results here.')
        
        # parameter for swin_unet
        parser.add_argument('--resume', type=str, default="", help='')
        parser.add_argument(
            '--cfg', type=str, default="./swin_unet/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
        parser.add_argument('--patch_size', type=list,  default=[256, 256],
                            help='patch size of network input')
        parser.add_argument('--zip', action='store_true',
                            help='use zipped dataset instead of folder dataset')
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--accumulation-steps', type=int,
                            help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
        self.isTrain = True
        return parser
