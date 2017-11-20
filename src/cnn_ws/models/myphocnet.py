import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from cnn_ws.spatial_pyramid_layers.gpp import GPP


class PHOCNet(nn.Module):
    '''
    Network class for generating PHOCNet and TPP-PHOCNet architectures
    '''

    def __init__(self, n_out, input_channels=1, gpp_type='spp', pooling_levels=3, pool_type='max_pool'):
        super(PHOCNet, self).__init__()
        # some sanity checks
        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')
        # set up Conv Layers
        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # create the spatial pooling layer
        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.fc5 = nn.Linear(pooling_output_size, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, n_out)

    def forward(self, x):
        y = F.relu(self.conv1_1(x))
        y = F.relu(self.conv1_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv3_1(y))
        y = F.relu(self.conv3_2(y))
        y = F.relu(self.conv3_3(y))
        y = F.relu(self.conv3_4(y))
        y = F.relu(self.conv3_5(y))
        y = F.relu(self.conv3_6(y))
        y = F.relu(self.conv4_1(y))
        y = F.relu(self.conv4_2(y))
        y = F.relu(self.conv4_3(y))

        y = self.pooling_layer_fn.forward(y)
        y = F.relu(self.fc5(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.fc6(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.fc7(y)
        return y

    def init_weights(self):
        self.apply(PHOCNet._init_weights_he)


    '''
    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #nn.init.kaiming_normal(m.weight.data)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**(1/2.0))
            if hasattr(m, 'bias'):
                nn.init.constant(m.bias.data, 0)
    '''

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            #nn.init.kaiming_normal(m.weight.data)
            nn.init.constant(m.bias.data, 0)