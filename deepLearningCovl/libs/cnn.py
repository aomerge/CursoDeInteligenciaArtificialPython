import torch  # type: ignore
class CNN(torch.nn.Module):
    """
    Esta clase define una red neuronal convolucional
    @param obs_shape: tuple: Dimensiones de la observación
    @param output_shape: int: Número de acciones
    """
    def __init__(self, obs_shape, output_shape, device = "cpu"):
        super(CNN, self).__init__()
        self.device = device
        ## input_shape  cx84x84
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = obs_shape[0], out_channels = 64, kernel_size = 8, stride = 4, padding = 1),
            torch.nn.ReLU()
            )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 0),
            torch.nn.ReLU()
            )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0),
            torch.nn.ReLU()
            )
        
        self.out = torch.nn.Linear(18*18*32, output_shape)
        
    def forward(self, x):
        """
        Esta función se encarga de propagar la red
        @param x: torch.Tensor: Tensor de entrada
        @return torch.Tensor: Tensor de salida
        """
        x = torch.from_numpy(x).float().to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)