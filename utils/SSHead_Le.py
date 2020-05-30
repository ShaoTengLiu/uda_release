from torch import nn
import torch.nn.functional as F

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

def extractor_from_layer3_Le(net):
    class lenet(nn.Module):
        def __init__(self):
            """Init LeNet encoder."""
            super(lenet, self).__init__()
            self.encoder = net.encoder
            self.fc1 = net.fc1
            self.bn3 = net.bn3
        def forward(self, input):
            """Forward the LeNet."""
            conv_out = self.encoder(input)
            feat = self.fc1(conv_out.view(-1, 50 * 5 * 5))
            feat = self.bn3(feat)
            return feat
    result = lenet()
    return result

def linear_on_layer3_Le(classes):
    class lenet_fc(nn.Module):
        def __init__(self):
            """Init LeNet encoder."""
            super(lenet_fc, self).__init__()
            self.fc2 = nn.Linear(500, classes)
        def forward(self, input):
            """Forward the LeNet."""
            feat = F.dropout(F.relu(input), training=self.training)
            feat = self.fc2(feat)
            return feat
    result = lenet_fc()
    return result
