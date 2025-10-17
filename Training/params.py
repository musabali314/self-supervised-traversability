class Params:
    def __init__(self):
        self.pretrained = True
        self.output_size = (424, 240)  # W x H
        self.output_channels = 1
        self.bottleneck_dim = 512
        self.train_csv = "../file.csv"  # adjust path if needed
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 50
        self.device = 'cuda'  # or 'cpu'
