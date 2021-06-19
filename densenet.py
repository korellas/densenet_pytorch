# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TransitionBlock(nn.Module):
    def __init__(self, in_channels: int, compression_factor=0.5):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(in_channels * compression_factor),
                              kernel_size=1,
                              stride=1,
                              )
        self.pooling = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        x = self.pooling(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super(ConvBlock, self).__init__()
        self.bn_1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=growth_rate * 4,
                                kernel_size=1,
                                stride=1)
        self.bn_2 = nn.BatchNorm2d(num_features=self.conv_1.out_channels)
        self.conv_2 = nn.Conv2d(in_channels=self.conv_1.out_channels,
                                out_channels=growth_rate,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.out_channels = in_channels + self.conv_2.out_channels

    def forward(self, x):
        out = F.relu(self.bn_1(x))
        out = self.conv_1(out)
        out = F.relu(self.bn_2(out))
        out = self.conv_2(out)
        # print(x.shape, out.shape)
        out = torch.cat((out, x), 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        # self.convblock = ConvBlock(in_channels=in_channels, growth_rate=growth_rate)
        self.convblocks = nn.ModuleList()
        for i in range(num_layers):
            block = ConvBlock(in_channels=in_channels, growth_rate=growth_rate)
            in_channels = block.out_channels
            self.convblocks.append(block)
        # print(self.convblocks)

    def forward(self, x):
        # x = self.convblock(x)
        for layer in self.convblocks:
            x = layer(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels: int, dense_layers: list, growth_rate: int,
                 num_classes: int, compression_factor=0.5):
        super(DenseNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=growth_rate * 2,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn = nn.BatchNorm2d(num_features=self.conv.out_channels)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.denseblocks = nn.ModuleList()
        num_channels = self.conv.out_channels
        last_i = len(dense_layers)
        for i, num_layers in enumerate(dense_layers):
            self.denseblocks.append(
                DenseBlock(num_channels, num_layers, growth_rate)
            )
            num_channels += num_layers * growth_rate
            if i != last_i:
                self.denseblocks.append(
                    TransitionBlock(in_channels=num_channels, compression_factor=compression_factor)
                )
                num_channels = int(num_channels * compression_factor)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        for block in self.denseblocks:
            x = block(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = DenseNet(3, [6, 12, 8], 32, 10, 0.5)
    device = torch.device('cuda')
    net.to(device)
    print(net)
    x = torch.randn((16, 3, 224, 224)).to(device) #pytorch에서는 batch, channel, Height, Width
    y = net(x)
    print(y.shape)

    target = torch.randn_like(y)
    criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    y = net(x)
    loss = criterion(y, target)
    print(loss)
    loss.backward()
    optimizer.step()
    y = net(x)
    loss = criterion(y, target)
    print(loss)
    list(net.parameters())
# %%
