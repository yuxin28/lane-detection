import torch
import torch.nn as nn
import torch.nn.functional as F


## TO DO 1: For each of the modules given below complete the implementation
# using the figure and table given in the task pdf document.

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2 times"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        # Step 3a
        self.d_conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())

    def forward(self, x):
        x = self.d_conv(x)
        # Step 3a
        return x


class encoder(nn.Module):
    '''(maxpool => double_conv)'''

    def __init__(self, in_ch, out_ch):
        super(encoder, self).__init__()
        # Step 3b
        self.encoder = nn.Sequential(nn.MaxPool2d(2, 2),
                                     double_conv(in_ch, out_ch))

    def forward(self, x):
        # Step 3b
        x = self.encoder(x)
        return x


class decoder(nn.Module):
    """(up_conv x1 => concatenate with x2 => double_conv)
    x1: tensor output from previous layer (from below)
    x2: tensor output from encoder layer at same resolution level (from left)
    """

    def __init__(self, in_ch, out_ch):
        super(decoder, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels=in_ch, stride=2, out_channels=out_ch, kernel_size=2)
        self.d_conv = double_conv(in_ch, out_ch).d_conv
        # Step 3c

    def forward(self, x1, x2):
        # Step 3c: Remove below line  x = None and complete the implementation
        x1 = self.tran_conv(x1)
        x = torch.cat((x1, x2), 1)
        x = self.d_conv(x)

        return x


class output_module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(output_module, self).__init__()
        # Step 3d
        self.output = nn.Conv2d(in_channels=in_ch, out_channels=2, kernel_size=1, )

    def forward(self, x):
        # Step 3d
        x = self.output(x)
        return x


## TO DO 2: Using the modules defined above, construct the complete U-Net
# architecture.

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        # Step 3e
        self.double_conv = double_conv(in_ch=n_channels, out_ch=64)
        self.end1 = encoder(64, 128)
        self.end2 = encoder(128, 256)
        self.end3 = encoder(256, 512)
        self.end4 = encoder(512, 1024)
        self.decode1 = decoder(1024, 512)
        self.decode2 = decoder(512, 256)
        self.decode3 = decoder(256, 128)
        self.decode4 = decoder(128, 64)
        self.output = output_module(64, n_classes)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.end1(x1)
        x3 = self.end2(x2)
        x4 = self.end3(x3)
        x5 = self.end4(x4)
        f1 = self.decode1(x5, x4)
        f2 = self.decode2(f1, x3)
        f3 = self.decode3(f2, x2)
        f4 = self.decode4(f3, x1)
        x = self.output(f4)
        return x


## TO DO 3: Implement a network prediction function using the Pytorch
# softmax layer.

def get_network_prediction(network_output):
    """
    Using softmax on network output to get final prediction and prediction
    probability for the lane class.

    The input will have 4 dimension: N x C x H x W , where N: no of samples
    in mini-batch. This is defined as batch_size in the dataloader (see notebook).
    Recall from before C: Channels, H: Height, W: Width

    Both output tensors, i.e., predicted_labels and lane_probability will have
    3 dimensions: N x H X W

    """
    ## Step 3f: Delete the lines below and complete the implementation.
#     print(network_output)
    network_output = nn.Softmax(1)(network_output)
#     predicted_labels = network_output[:, 0]
#     print(predicted_labels)
#     lane_probability = network_output[:, 1]
#     print(lane_probability)
    # Ensure that the probability tensor does not have the channel dimension
    lane_probability = network_output[:, 1]
    predicted_labels = (lane_probability >= 0.5)

    return predicted_labels, lane_probability
