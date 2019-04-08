
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from scipy.ndimage.interpolation import rotate as sc_rotate

class InterpretableAE(nn.Module):
    def __init__(self, height, width, device, latent_dim=16):
        super(InterpretableAE, self).__init__()

        self.width  = width
        self.height = height
        self.device = device

        self.n_layers = 3
        self.feat_sz  = (self.height // 2**self.n_layers) * (self.width // 2**self.n_layers)
        
        filter_arr = [8, 16, 16]#, 16, 16]
        self.max_channels = filter_arr[-1]
        
        # Init model layers
        self.down1 = nn.Conv2d(           1,  filter_arr[0], 3, stride=2,padding=1) #shape //2
        self.down2 = nn.Conv2d(filter_arr[0], filter_arr[1], 3, stride=2,padding=1) #shape //2
        self.down3 = nn.Conv2d(filter_arr[1], filter_arr[2], 3, stride=2,padding=1) #shape //2
        #self.down4 = nn.Conv2d(filter_arr[2], filter_arr[3], 3, stride=2,padding=1) #shape //2
        #self.down5 = nn.Conv2d(filter_arr[3], filter_arr[4], 3, stride=2,padding=1) #shape //2
        
        self.latentd = nn.Linear(self.max_channels * self.feat_sz, latent_dim)
        #decoder
        self.latentu = nn.Linear(latent_dim, self.feat_sz * self.max_channels)
        
        other_opts = {'stride':2, 'padding':1, 'output_padding':1}
        
        #self.up5   = nn.ConvTranspose2d(filter_arr[4], filter_arr[3], 3, **other_opts)
        #self.up4   = nn.ConvTranspose2d(filter_arr[3], filter_arr[2], 3, **other_opts)
        self.up3   = nn.ConvTranspose2d(filter_arr[2], filter_arr[1], 3, **other_opts)
        self.up2   = nn.ConvTranspose2d(filter_arr[1], filter_arr[0], 3, **other_opts)
        self.up1   = nn.ConvTranspose2d(filter_arr[0],             1, 3, **other_opts)
        
    def encode(self, x, params):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = F.relu(self.down3(x))
        #x = F.relu(self.down4(x))
        #x = F.relu(self.down5(x))
        x =  x.view(-1, self.max_channels * self.feat_sz) #flatten
        x = self.latentd(x)
        x = self.feature_transformer(x, params)
        return x
    
    def decode(self, x):
        x = self.latentu(x)
        x = x.view(-1, self.max_channels, (self.height // 2**self.n_layers), (self.width // 2**self.n_layers))
        #x = F.relu(self.up5(x))
        #x = F.relu(self.up4(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up2(x))
        x = torch.sigmoid(self.up1(x))
        return x

    def forward(self, x, params):
        return self.decode(self.encode(x, params))
    
    def feature_transformer(self, input, params):
        """For now we assume the params are just a single rotation angle

        Args:
            input: [N,c] tensor, where c = 2*int
            params: [N,1] tensor, with values in [0,2*pi)
        Returns:
            [N,c] tensor
        """
        # First reshape activations into [N,c/2,2,1] matrices
        x = input.view(input.size(0), input.size(1)//2 , 2 ,1)
        # Construct the transformation matrix
        sin = torch.sin(params)
        cos = torch.cos(params)
        transform = torch.cat([sin, -cos, cos, sin], 1)
        transform = transform.view(transform.size(0),1,2,2).to(self.device)
        # Multiply: broadcasting taken care of automatically
        # [N,1,2,2] @ [N,channels/2,2,1]
        output = torch.matmul(transform, x)
        # Reshape and return
        return output.view(input.size())

def rotate_tensor(input):
    """Nasty hack to rotate images in a minibatch, this should be parallelized
    and set in PyTorch

    Args:
        input: [N,c,h,w] **numpy** tensor
    Returns:
        rotated output and angles in radians
    """
    angles = 2*np.pi*np.random.rand(input.shape[0])
    angles = angles.astype(np.float32)
    outputs = []
    for i in range(input.shape[0]):
        output = sc_rotate(input[i,...], 180*angles[i]/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)
    return np.stack(outputs, 0), angles

def random_rotate(input):
    angle = 2*np.pi*np.random.rand().astype(np.float32)
    for i in range(input.shape[0]):
        output = sc_rotate(input[i,...], 180 * angle/np.pi, axes=(1,2), reshape=False)
        outputs.append(output)
    return np.stack(outputs, 0)

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # Reshape data
        targets, angles = rotate_tensor(data.numpy())
        targets = torch.from_numpy(targets).to(device, dtype=torch.float)
        angles = torch.from_numpy(angles).to(device)
        angles = angles.view(angles.size(0), 1)

        # Forward pass
        data = data.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data, angles)

        # Binary cross entropy loss
        loss_fnc = nn.BCELoss(reduction='sum')
        loss = loss_fnc(output, targets)

        # Backprop
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()
