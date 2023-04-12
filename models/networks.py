import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from .sync_batchnorm import SynchronizedBatchNorm2d
import re
from .AdaIN.function import adaptive_instance_normalization as FAdaIN

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(opt, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'ff':
        net = FFGenerator(opt, input_nc, output_nc, ngf)
    elif netG == 'pff':
        net = pFFGenerator(opt, input_nc, output_nc, ngf)
    elif netG == 'cycle':
        net = CycleGenerator(opt, input_nc, output_nc, ngf)
    elif netG == 'rff':
        net = rFFGenerator(opt, input_nc, output_nc, ngf)
    elif netG == 'tsit':
        net = TSITGenerator(opt, input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

        elif gan_mode == 'hinge':
            self.loss = None

        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real, for_discriminator):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if for_discriminator==True and self.fake_label==0:
            self.fake_label=self.fake_label*0
        else:
            self.fake_label=self.fake_label*0 
   

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator = True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real, for_discriminator)
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'hinge':
            if for_discriminator:
                target_tensor = self.get_target_tensor(prediction, False, for_discriminator)
                if target_is_real:
                    minval = torch.min(prediction - 1, target_tensor)
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-prediction - 1, target_tensor, for_discriminator)
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(prediction)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0  / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
         loss = 0
         for i in range(len(x_vgg)):
              loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
         return loss

class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
         return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        
        result = [input]
        result.append(self.model[:2](result[-1]))
        result.append(self.model[2:5](result[-1]))
        result.append(self.model[5:8](result[-1]))
        result.append(self.model[8:](result[-1]))        
        
        return result[1:]


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
#............................................................. working: 2022_04_30 .................................................................................................
class SelfAttention(nn.Module):
    """Self-attention GANにおけるSelf-attention
    ただしSAGANのSpectral Normを抜いているので注意

    Arguments:
        dims {int} -- 4Dテンソルの入力チャンネル
    """    
    def __init__(self, dims):
        super().__init__()
        self.conv_theta = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_phi = nn.Conv2d(dims, dims // 8, kernel_size=1)
        self.conv_g = nn.Conv2d(dims, dims // 2, kernel_size=1)
        self.conv_attn = nn.Conv2d(dims // 2, dims, kernel_size=1)
        self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        batch, ch, height, width = inputs.size()
        # theta path
        theta = self.conv_theta(inputs)
        theta = theta.view(batch, ch // 8, height * width).permute([0, 2, 1])  # (B, HW, C/8)        
        # phi path
        phi = self.conv_phi(inputs)
        phi = F.max_pool2d(phi, kernel_size=2)  # (B, C/8, H/2, W/2)
        phi = phi.view(batch, ch // 8, height * width // 4)  # (B, C/8, HW/4)
        # attention
        attn = torch.bmm(theta, phi)  # (B, HW, HW/4)
        attn = F.softmax(attn, dim=-1)
        # g path
        g = self.conv_g(inputs)
        g = F.max_pool2d(g, kernel_size=2)  # (B, C/2, H/2, W/2)
        g = g.view(batch, ch // 2, height * width // 4).permute([0, 2, 1])  # (B, HW/4, C/2)

        attn_g = torch.bmm(attn, g)  # (B, HW, C/2)
        attn_g = attn_g.permute([0, 2, 1]).view(batch, ch // 2, height, width)  # (B, C/2, H, W)
        attn_g = self.conv_attn(attn_g)
        return inputs + self.sigma_ratio * attn_g

class FFGenerator(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64):
        super().__init__()
        norm_en = nn.InstanceNorm2d
        
        self.en0 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.en0 = nn.Sequential(*self.en0)
        self.en1 = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*2),
                    nn.ReLU(True)]
        self.en1 = nn.Sequential(*self.en1)
        self.en2 = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*4),
                    nn.ReLU(True)]
        self.en2 = nn.Sequential(*self.en2)
        
        self.resen0 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen1 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen2 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen3 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen4 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resde0 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde1 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde2 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde3 = FADEResnetBlock(ngf*4, ngf*4, opt)
        
        fade_config_str = opt.norm_G.replace('spectral','')
        print("..............................................................................", opt.norm_G)
        self.fade0 = FADE(fade_config_str, ngf*4, ngf*4)
        self.fade1 = FADE(fade_config_str, ngf*2, ngf*2)
        
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de0 = nn.Sequential(*self.de0)
        self.de1 = [nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de1 = nn.Sequential(*self.de1)
        self.de2 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]
        self.de2 = nn.Sequential(*self.de2)
        self.act = nn.ReLU(True)

        #attention module
        self.self_resen0 = SelfAttention(256)
        self.self_resen1 = SelfAttention(256)
        self.self_resen2 = SelfAttention(256)
        self.self_resen3 = SelfAttention(256)
        self.self_resen4 = SelfAttention(256)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
        
    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def forward(self, input, real, netG, encode):
        content = input
        style = real
        encoder = netG #generator
        
        e0 = self.en0(content)
        e1 = self.en1(e0)
        e2 = self.en2(e1)
        
        re0 = self.self_resen0(self.resen0(e2))#; print("re0: ", re0.size())
        re1 = self.self_resen1(self.resen1(re0))#; print("re1: ", re1.size())
        re2 = self.self_resen2(self.resen2(re1))#; print("re2: ", re2.size())
        re3 = self.self_resen3(self.resen3(re2))#; print("re3: ", re3.size())
        re4 = self.self_resen4(self.resen4(re3))#; print("re4: ", re4.size())
        
        if encode == True:
           return e0, e1, e2, re0, re1, re2, re3
        else:
            sf0, sf1, sf2, sf3, sf4, sf5, sf6 = encoder(style, content, netG, True)
            
            x = self.fadain_alpha(re4, sf6, alpha = 1)
            x = self.resde0(x, re3)
            x = self.fadain_alpha(x, sf5, alpha = 1)
            x = self.resde1(x, re2)
            x = self.fadain_alpha(x, sf4, alpha = 1)
            x = self.resde2(x, re1)
            x = self.fadain_alpha(x, sf3, alpha = 1)
            x = self.resde3(x, re0)
            
            x = self.fadain_alpha(x, sf2, alpha = 1)
            x = self.fade0(x, e2)
            x = self.de0(self.act(x))
            x = self.fadain_alpha(x, sf1, alpha = 1)
            x = self.fade1(x, e1)
            x = self.de1(self.act(x))
            x = self.fadain_alpha(x, sf0, alpha = 1)
            x = self.de2(self.act(x))
            
            return x 

class pFFGenerator(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64):
        super().__init__()
        norm_en = nn.InstanceNorm2d
        
        self.en0 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.en0 = nn.Sequential(*self.en0)
        self.en1 = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*2),
                    nn.ReLU(True)]
        self.en1 = nn.Sequential(*self.en1)
        self.en2 = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*4),
                    nn.ReLU(True)]
        self.en2 = nn.Sequential(*self.en2)
        
        self.resen0 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen1 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen2 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen3 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen4 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen5 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen6 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen7 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen8 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)

        
        fade_config_str = opt.norm_G.replace('spectral','')
        self.fade0 = FADE(fade_config_str, ngf*4, ngf*4)
        
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de0 = nn.Sequential(*self.de0)
        self.de1 = [nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)]
        self.de1 = nn.Sequential(*self.de1)
        self.de2 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]
        self.de2 = nn.Sequential(*self.de2)
        self.act = nn.ReLU(True)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
        
    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def forward(self, input, real, netG, encode):
        content = input
        style = real
        encoder = netG
        
        e0 = self.en0(content)
        e1 = self.en1(e0)
        e2 = self.en2(e1)
        
        re0 = self.resen0(e2)
        re1 = self.resen1(re0)
        re2 = self.resen2(re1)
        re3 = self.resen3(re2)
        re4 = self.resen4(re3)
        
        if encode == True:
           return e0, e1, e2
        else:
            sf0, sf1, sf2 = encoder(style, content, netG, True)
            
            x = self.resen5(re3)
            x = self.resen6(re3)
            x = self.resen7(re3)
            x = self.resen8(re3)
            
            x = self.fadain_alpha(x, sf2, alpha = 1)
            x = self.fade0(x, e2)
            x = self.de0(self.act(x))
            x = self.fadain_alpha(x, sf1, alpha = 1)
            x = self.de1(self.act(x))
            x = self.fadain_alpha(x, sf0, alpha = 1)
            x = self.de2(self.act(x))
            
            return x 

class CycleGenerator(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64):
        super().__init__()
        norm_en = nn.InstanceNorm2d
        
        self.en0 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.en0 = nn.Sequential(*self.en0)
        self.en1 = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*2),
                    nn.ReLU(True)]
        self.en1 = nn.Sequential(*self.en1)
        self.en2 = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*4),
                    nn.ReLU(True)]
        self.en2 = nn.Sequential(*self.en2)
        
        self.resen0 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen1 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen2 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen3 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen4 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen5 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen6 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen7 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen8 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resde = FADEResnetBlock(ngf*4, ngf*4, opt)
        
        fade_config_str = opt.norm_G.replace('spectral','')
        self.fade0 = FADE(fade_config_str, ngf*4, ngf*4)
        self.fade1 = FADE(fade_config_str, ngf*2, ngf*2)
        
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True),
                   norm_en(ngf*2),
                   nn.ReLU(True)]
        self.de0 = nn.Sequential(*self.de0)
        self.de1 = [nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.de1 = nn.Sequential(*self.de1)
        self.de2 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]
        self.de2 = nn.Sequential(*self.de2)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
        
    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def forward(self, input, real, netG, encode):
        content = input
        style = real
        encoder = netG
        
        e0 = self.en0(content)
        e1 = self.en1(e0)
        e2 = self.en2(e1)
        
        re0 = self.resen0(e2)
        re1 = self.resen1(re0)
        re2 = self.resen2(re1)
        re3 = self.resen3(re2)
        re4 = self.resen4(re3)
        
        if encode == True:
           return e0, e1, e2 
        else:
            sf0, sf1, sf2 = encoder(style, content, netG, True)
            
            
            x = self.resen5(re4)
            
            x = self.resen6(x)
            
            x = self.resen7(x)
            
            x = self.resen8(x)
            

            x = self.de0(x)

            
            x = self.de1(x)

            x = self.de2(x)
            
            return x 

class rFFGenerator(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64):
        super().__init__()
        norm_en = nn.InstanceNorm2d
        
        self.en0 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.en0 = nn.Sequential(*self.en0)
        self.en1 = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*2),
                    nn.ReLU(True)]
        self.en1 = nn.Sequential(*self.en1)
        self.en2 = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=True),
                    norm_en(ngf*4),
                    nn.ReLU(True)]
        self.en2 = nn.Sequential(*self.en2)
        
        self.resen0 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen1 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen2 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen3 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resen4 = ResnetBlock(ngf*4, padding_type='reflect', norm_layer=norm_en,
                                  use_dropout=False, use_bias=True)
        self.resde0 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde1 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde2 = FADEResnetBlock(ngf*4, ngf*4, opt)
        self.resde3 = FADEResnetBlock(ngf*4, ngf*4, opt)
        
        
        self.de0 = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True),
                    norm_en(ngf*2),
                    nn.ReLU(True)]
        self.de0 = nn.Sequential(*self.de0)
        self.de1 = [nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True),
                    norm_en(ngf),
                    nn.ReLU(True)]
        self.de1 = nn.Sequential(*self.de1)
        self.de2 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]
        self.de2 = nn.Sequential(*self.de2)
        self.act = nn.ReLU(True)


        #attention module
        self.self_resen0 = SelfAttention(256)
        self.self_resen1 = SelfAttention(256)
        self.self_resen2 = SelfAttention(256)
        self.self_resen3 = SelfAttention(256)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
        
    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def forward(self, input, real, netG, encode):
        content = input
        style = real
        encoder = netG
        
        e0 = self.en0(content)
        e1 = self.en1(e0)
        e2 = self.en2(e1)
        
        re0 = self.resen0(e2)
        re1 = self.resen1(re0)
        re2 = self.resen2(re1)
        re3 = self.resen3(re2)
        re4 = self.resen4(re3)
        
        if encode == True:
           return re0, re1, re2, re3
        else:
           sf3, sf4, sf5, sf6 = encoder(style, content, netG, True)
            
           x = self.fadain_alpha(re4, sf6, alpha = 1)
           x = self.resde0(x, self.self_resen3(re3))
           x = self.fadain_alpha(x, sf5, alpha = 1)
           x = self.resde1(x, self.self_resen2(re2))
           x = self.fadain_alpha(x, sf4, alpha = 1)
           x = self.resde2(x, self.self_resen1(re1))
           x = self.fadain_alpha(x, sf3, alpha = 1)
           x = self.resde3(x, self.self_resen0(re0))
            

           x = self.de0(x)
           x = self.de1(x)
           x = self.de2(x)
            
           return x

class TSITGenerator(nn.Module):
    def __init__(self, opt, input_nc, output_nc, ngf=64):
        super().__init__()
        self.res_0 = StreamResnetBlock(input_nc, 1 * ngf, opt)  # 64-ch feature
        self.res_1 = StreamResnetBlock(1  * ngf, 2  * ngf, opt)   # 128-ch  feature
        self.res_2 = StreamResnetBlock(2  * ngf, 4  * ngf, opt)   # 256-ch  feature
        self.res_3 = StreamResnetBlock(4  * ngf, 8  * ngf, opt)   # 512-ch  feature
        self.res_4 = StreamResnetBlock(8  * ngf, 16 * ngf, opt)   # 1024-ch feature

        self.up_0 = FADEResnetBlock(16 * ngf, 8 * ngf, opt)
        self.up_1 = FADEResnetBlock(8 * ngf, 4 * ngf, opt)
        self.up_2 = FADEResnetBlock(4 * ngf, 2 * ngf, opt)
        self.up_3 = FADEResnetBlock(2 * ngf, 1 * ngf, opt)

        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)



    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

        
    def fadain_alpha(self, content_feat, style_feat, alpha=1.0, c_mask=None, s_mask=None):
        assert 0 <= alpha <= 1
        t = FAdaIN(content_feat, style_feat, c_mask, s_mask)
        t = alpha * t + (1 - alpha) * content_feat
        return t
    
    def forward(self, input, real, netG, encode):
        content = input
        style = real
        encoder = netG
        
        ft0 = self.res_0(content) # (n,64,256,512)

        ft1 = self.down(ft0)
        ft1 = self.res_1(ft1)    # (n,128,128,256)

        ft2 = self.down(ft1)
        ft2 = self.res_2(ft2)    # (n,256,64,128)

        ft3 = self.down(ft2)
        ft3 = self.res_3(ft3)    # (n,512,32,64)

        ft4 = self.down(ft3)
        ft4 = self.res_4(ft4)    # (n,1024,16,32)


        
        if encode == True:
           return ft0, ft1, ft2, ft3, ft4
        else:
           sft0, sft1, sft2, sft3, sft4 = encoder(style, content, netG, True)

           x = self.fadain_alpha(ft4, sft4, alpha=1.0)
           x = self.up_0(x, ft4)

           x = self.up(x)
           x = self.fadain_alpha(x, sft3, alpha=1.0) 
           x = self.up_1(x, ft3)

           x = self.up(x)
           x = self.fadain_alpha(x, sft2, alpha=1.0) 
           x = self.up_2(x, ft2)

           x = self.up(x)
           x = self.fadain_alpha(x, sft1, alpha=1.0) 
           x = self.up_3(x, ft1)

           x = self.up(x)
           x = self.conv_img(F.leaky_relu(x, 2e-1))
           x = F.tanh(x)
             
           return x

        
class FADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fin

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, bias=True)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        #print(opt.norm_G)
        fade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = FADE(fade_config_str, fin, fin)
        self.norm_1 = FADE(fade_config_str, fmiddle, fmiddle)
        if self.learned_shortcut:
            self.norm_s = FADE(fade_config_str, fin, fin)

    # Note the resnet block with FADE also takes in |feat|,
    # the feature map as input
    def forward(self, x, feat):
        x_s = self.shortcut(x, feat)

        dx = self.conv_0(self.actvn(self.norm_0(x, feat)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, feat)))

        out = x_s + dx

        return out

    def shortcut(self, x, feat):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, feat))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
   
class StreamResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fin

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_S:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        subnorm_type = opt.norm_S.replace('spectral', '')
        if subnorm_type == 'batch':
            self.norm_layer_in = nn.BatchNorm2d(fin, affine=True)
            self.norm_layer_out= nn.BatchNorm2d(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = nn.BatchNorm2d(fout, affine=True)
        elif subnorm_type == 'syncbatch':
            self.norm_layer_in = SynchronizedBatchNorm2d(fin, affine=True)
            self.norm_layer_out= SynchronizedBatchNorm2d(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = SynchronizedBatchNorm2d(fout, affine=True)
        elif subnorm_type == 'instance':
            self.norm_layer_in = nn.InstanceNorm2d(fin, affine=False)
            self.norm_layer_out= nn.InstanceNorm2d(fout, affine=False)
            if self.learned_shortcut:
                self.norm_layer_s = nn.InstanceNorm2d(fout, affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.actvn(self.norm_layer_in(self.conv_0(x)))
        dx = self.actvn(self.norm_layer_out(self.conv_1(dx)))

        out = x_s + dx

        return out

    def shortcut(self,x):
        if self.learned_shortcut:
            x_s = self.actvn(self.norm_layer_s(self.conv_s(x)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
        

    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_feature = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_feature[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_feature[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_feature[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_feature[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_feature[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
        
class FADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        print(config_text)
        assert config_text.startswith('fade') #while training is used
        parsed = re.search('fade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in FADE'
                             % param_free_norm_type)

        pw = ks // 2
        self.mlp_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, feat):
        # Step 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Step 2. produce scale and bias conditioned on feature map
        gamma = self.mlp_gamma(feat)
        beta = self.mlp_beta(feat)

        # Step 3. apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out