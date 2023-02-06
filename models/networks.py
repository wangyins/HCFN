import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, bias_input_nc, output_nc, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = ColorNet(input_nc, bias_input_nc, output_nc, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)


class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        conv_block = [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class global_network(nn.Module):
    def __init__(self, in_dim):
        super(global_network, self).__init__()
        model = [nn.Conv2d(in_dim, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        model += [nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        self.model_1 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.ReLU(True)])
        self.model_2 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.ReLU(True)])
        self.model_3 = nn.Sequential(*[nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.ReLU(True)])

    def forward(self, x):
        x = self.model(x)
        x1 = self.model_1(x)
        x2 = self.model_2(x)
        x3 = self.model_3(x)

        return x1, x2, x3


class ref_network(nn.Module):
    def __init__(self, norm_layer):
        super(ref_network, self).__init__()
        model1 = [nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(64)]
        self.model1 = nn.Sequential(*model1)
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(128)]
        self.model2 = nn.Sequential(*model2)
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(128)]
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(256)]
        self.model4 = nn.Sequential(*model4)

    def forward(self, color, corr, H, W, mode):
        conv1 = self.model1(color[:,:,::2,::2])
        conv2 = self.model2(conv1[:,:,::2,::2])

        assert mode in {'refer', 'pre'}
        if mode == 'refer':
            conv2_flatten = conv2.view(conv2.shape[0], conv2.shape[1], -1)
            align = torch.bmm(conv2_flatten, corr)
        elif mode == 'pre':
            conv2_pad = F.pad(conv2, (6, 6, 6, 6), mode='replicate')
            conv2_uf = F.unfold(conv2_pad, kernel_size=13)
            conv2_uf = conv2_uf.view([conv2.shape[0], conv2.shape[1], 13*13, H*W])
            corr = corr.unsqueeze(1)
            align = (corr * conv2_uf).sum(2)
        align_1 = align.view(align.shape[0], align.shape[1], H, W)
        align_2 = self.model3(align_1[:,:,::2,::2])
        align_3 = self.model4(align_2[:,:,::2,::2])

        return align_1, align_2, align_3


class conf_feature(nn.Module):
    def __init__(self):
        super(conf_feature, self).__init__()
        self.fc1 = nn.Sequential(*[nn.Conv1d(4096, 1024, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(True)])
        self.fc2 = nn.Sequential(*[nn.Conv1d(1024, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid()])
        self.fcp = nn.Sequential(*[nn.Conv1d(13*13, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid()])

    def forward(self, x, mode):
        assert mode in {'refer', 'pre'}
        if mode == 'refer':
            x = self.fc1(x)
            x = self.fc2(x)
        elif mode == 'pre':
            x = self.fcp(x)
        return x


class ColorNet(nn.Module):
    def __init__(self, input_nc, bias_input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True):
        super(ColorNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        use_bias = True

        model_head = [nn.Conv2d(input_nc, 32, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.ReLU(True),
                      norm_layer(32)]

        model1 = [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        # ResBlock0
        resblock0_1 = [nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias), norm_layer(512), nn.ReLU(True)]
        self.resblock0_2 = ResBlock(512, norm_layer, False, use_bias)
        self.resblock0_3 = ResBlock(512, norm_layer, False, use_bias)

        # Conv8
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(256), ]

        # ResBlock1
        resblock1_1 = [nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias), norm_layer(256), nn.ReLU(True)]
        self.resblock1_2 = ResBlock(256, norm_layer, False, use_bias)
        self.resblock1_3 = ResBlock(256, norm_layer, False, use_bias)

        # Conv9
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model9 = [nn.ReLU(True), ]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(128), ]

        # ResBlock2
        resblock2_1 = [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), norm_layer(128), nn.ReLU(True)]
        self.resblock2_2 = ResBlock(128, norm_layer, False, use_bias)
        self.resblock2_3 = ResBlock(128, norm_layer, False, use_bias)

        # Conv10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model10 = [nn.ReLU(True), ]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), ]
        model10 += [nn.LeakyReLU(negative_slope=.2), ]

        # Conv Global
        self.global_network = global_network(bias_input_nc)

        # Conv Ref
        self.ref_network = ref_network(norm_layer)

        # conf feature
        self.conf_feature = conf_feature()

        model_tail_1 = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
                        nn.LeakyReLU(negative_slope=.2)]
        model_tail_2 = [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                        nn.LeakyReLU(negative_slope=.2)]
        model_tail_3 = [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),
                        nn.LeakyReLU(negative_slope=.2)]

        model_out1 = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]
        model_out2 = [nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]
        model_out3 = [nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.resblock0_1 = nn.Sequential(*resblock0_1)
        self.resblock1_1 = nn.Sequential(*resblock1_1)
        self.resblock2_1 = nn.Sequential(*resblock2_1)

        self.model_out1 = nn.Sequential(*model_out1)
        self.model_out2 = nn.Sequential(*model_out2)
        self.model_out3 = nn.Sequential(*model_out3)
        self.model_head = nn.Sequential(*model_head)
        self.model_tail_1 = nn.Sequential(*model_tail_1)
        self.model_tail_2 = nn.Sequential(*model_tail_2)
        self.model_tail_3 = nn.Sequential(*model_tail_3)
        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=13, stride=1, padding=0,
                                                             dilation=1)
        self.PRE_FRAME_ALPHA = 0.001

    def forward(self, input, bias_input, ref_input, ref_color, pre_input, pre_color):
        bias_input = bias_input.view(input.shape[0], -1, 1, 1)

        conv_head = self.model_head(input)
        in_1 = self.model1(conv_head[:, :, ::2, ::2])
        in_2 = self.model2(in_1[:, :, ::2, ::2])
        in_3 = self.model3(in_2[:, :, ::2, ::2])
        in_4 = self.model4(in_3[:, :, ::2, ::2])
        in_5 = self.model5(in_4)
        in_6 = self.model6(in_5)

        conv_head_ref = self.model_head(ref_input)
        ref_1 = self.model1(conv_head_ref[:, :, ::2, ::2])
        ref_2 = self.model2(ref_1[:, :, ::2, ::2])
        ref_3 = self.model3(ref_2[:, :, ::2, ::2])
        ref_4 = self.model4(ref_3[:, :, ::2, ::2])
        ref_5 = self.model5(ref_4)
        ref_6 = self.model6(ref_5)

        conv_head_pre = self.model_head(pre_input)
        pre_1 = self.model1(conv_head_pre[:, :, ::2, ::2])
        pre_2 = self.model2(pre_1[:, :, ::2, ::2])
        pre_3 = self.model3(pre_2[:, :, ::2, ::2])
        pre_4 = self.model4(pre_3[:, :, ::2, ::2])
        pre_5 = self.model5(pre_4)
        pre_6 = self.model6(pre_5)

        t1 = F.interpolate(in_1, scale_factor=0.5, mode='bilinear')
        t2 = in_2
        t3 = F.interpolate(in_3, scale_factor=2, mode='bilinear')
        t4 = F.interpolate(in_4, scale_factor=4, mode='bilinear')
        t5 = F.interpolate(in_5, scale_factor=4, mode='bilinear')
        t6 = F.interpolate(in_6, scale_factor=4, mode='bilinear')
        t = torch.cat((t1, t2, t3, t4, t5, t6), dim=1)

        r1 = F.interpolate(ref_1, scale_factor=0.5, mode='bilinear')
        r2 = ref_2
        r3 = F.interpolate(ref_3, scale_factor=2, mode='bilinear')
        r4 = F.interpolate(ref_4, scale_factor=4, mode='bilinear')
        r5 = F.interpolate(ref_5, scale_factor=4, mode='bilinear')
        r6 = F.interpolate(ref_6, scale_factor=4, mode='bilinear')
        r = torch.cat((r1, r2, r3, r4, r5, r6), dim=1)

        p1 = F.interpolate(pre_1, scale_factor=0.5, mode='bilinear')
        p2 = pre_2
        p3 = F.interpolate(pre_3, scale_factor=2, mode='bilinear')
        p4 = F.interpolate(pre_4, scale_factor=4, mode='bilinear')
        p5 = F.interpolate(pre_5, scale_factor=4, mode='bilinear')
        p6 = F.interpolate(pre_6, scale_factor=4, mode='bilinear')
        p = torch.cat((p1, p2, p3, p4, p5, p6), dim=1)

        input_T_flatten = t.view(t.shape[0], t.shape[1], -1).permute(0, 2, 1)
        input_R_flatten = r.view(r.shape[0], r.shape[1], -1).permute(0, 2, 1)
        input_T_flatten = input_T_flatten / torch.norm(input_T_flatten, p=2, dim=-1, keepdim=True)
        input_R_flatten = input_R_flatten / torch.norm(input_R_flatten, p=2, dim=-1, keepdim=True)
        corr = torch.bmm(input_R_flatten, input_T_flatten.permute(0, 2, 1))
        conf = self.conf_feature(corr, mode='refer')
        conf = conf.view(conf.shape[0], 1, t2.shape[2], t2.shape[3])
        conf_1 = conf
        conf_2 = conf_1[:, :, ::2, ::2]
        conf_3 = conf_2[:, :, ::2, ::2]
        corr = F.softmax(corr/0.01, dim=1)
        align_1, align_2, align_3 = self.ref_network(ref_color, corr, t2.shape[2], t2.shape[3], mode='refer')
        conv_global1, conv_global2, conv_global3 = self.global_network(bias_input)

        p_norm = p / torch.norm(p, p=2, dim=1, keepdim=True)
        t_norm = t / torch.norm(t, p=2, dim=1, keepdim=True)
        corr_P = self.correlation_sampler(t_norm, p_norm)
        corr_P[corr_P == 0.0] = 1e-5  # discount padding at edge for softmax
        corr_P = corr_P.view([corr_P.shape[0], 13*13, t2.shape[2]*t2.shape[3]])
        conf_P = self.conf_feature(corr_P, mode='pre')
        conf_P = conf_P.view(conf_P.shape[0], 1, t2.shape[2], t2.shape[3])
        conf_P1 = conf_P
        conf_P2 = conf_P1[:, :, ::2, ::2]
        conf_P3 = conf_P2[:, :, ::2, ::2]
        corr_P = F.softmax(corr_P/0.01, dim=1)
        align_P1, align_P2, align_P3 = self.ref_network(pre_color, corr_P, t2.shape[2], t2.shape[3], mode='pre')

        conf_P1 = conf_P1 * self.PRE_FRAME_ALPHA
        conf_P2 = conf_P2 * self.PRE_FRAME_ALPHA
        conf_P3 = conf_P3 * self.PRE_FRAME_ALPHA

        conv_global1_repeat = conv_global1.expand_as(in_6)
        conv6_3_global = in_6 + (conf_3 * align_3 + (1 - conf_3) * conv_global1_repeat) * (
                    1 - conf_P3) + conf_P3 * align_P3

        conv7_resblock1 = self.resblock0_1(conv6_3_global)
        conv7_resblock2 = self.resblock0_2(conv7_resblock1)
        conv7_resblock3 = self.resblock0_3(conv7_resblock2)
        conv8_up = self.model8up(conv7_resblock3) + self.model3short8(in_3)
        conv8_3 = self.model8(conv8_up)
        conv_tail_1 = self.model_tail_1(conv8_3)
        fake_img1 = self.model_out1(conv_tail_1)

        conv_global2_repeat = conv_global2.expand_as(conv8_3)
        conv8_3_global = conv8_3 + (conf_2 * align_2 + (1 - conf_2) * conv_global2_repeat) * (
                    1 - conf_P2) + conf_P2 * align_P2

        conv8_resblock1 = self.resblock1_1(conv8_3_global)
        conv8_resblock2 = self.resblock1_2(conv8_resblock1)
        conv8_resblock3 = self.resblock1_3(conv8_resblock2)
        conv9_up = self.model9up(conv8_resblock3) + self.model2short9(in_2)
        conv9_3 = self.model9(conv9_up)
        conv_tail_2 = self.model_tail_2(conv9_3)
        fake_img2 = self.model_out2(conv_tail_2)

        conv_global3_repeat = conv_global3.expand_as(conv9_3)
        conv9_3_global = conv9_3 + (conf_1 * align_1 + (1 - conf_1) * conv_global3_repeat) * (
                    1 - conf_P1) + conf_P1 * align_P1
        conv9_resblock1 = self.resblock2_1(conv9_3_global)
        conv9_resblock2 = self.resblock2_2(conv9_resblock1)
        conv9_resblock3 = self.resblock2_3(conv9_resblock2)
        conv10_up = self.model10up(conv9_resblock3) + self.model1short10(in_1)
        conv10_2 = self.model10(conv10_up)
        conv_tail_3 = self.model_tail_3(conv10_2)
        fake_img3 = self.model_out3(conv_tail_3)

        return [fake_img1, fake_img2, fake_img3]
