if __name__ == "__main__":
    
    # run tests
    import sys
    root = '/home/mitch/PythonProjects/Geo-PGML'
    sys.path.append(f'{root}/src')
    
    import torch
    import torch.nn.functional as F
    from torchinfo import summary
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # make dummy data batch
    batch_size = 32
    n_length = 1000
    n_chan_in = 3
    n_chan_out = 1
    input_data = torch.randn(size=(batch_size, n_length, n_chan_in)).to(device)
    
    from models.unet import Unet1D, Unet1D_HF
    from models.resnet import ResNet1D
    from models.fcnn import FCNN
    from models.tcn import TCN1D


    # fcnn
    hidden = [128, 64, 32, 64, 128]
    act = F.tanh
    model = FCNN(n_chan_in, n_chan_out, hidden, act).to(device)

    s = summary(model, input_data=input_data)
    print()
    print(model._get_name())
    print(s.input_size)
    print(s)


    # u-net
    kernel_size = 7
    n_filters = 32

    model = Unet1D(n_chan_in, n_filters, kernel_size=kernel_size).to(device)
    
    s = summary(model, input_data=input_data.mT) # expects (B, C, L)
    print()
    print(model._get_name())
    print(s.input_size)
    print(s)

    #u-net hugging face
    input = {'sample':input_data.mT, 'timestep':n_length}
    model = Unet1D_HF(sample_size=n_length, in_channels=n_chan_in+16, 
                        out_channels=n_chan_out, block_out_channels=(32,32,64) ).to(device)
    s = summary(model, input_data=input)
    print(model._get_name())
    print(s.input_size)
    print(s)
    

    # resnet
    n_block = 3
    downsample_gap = 6
    increasefilter_gap = 12
    base_filters = 32
    kernel_size = 3
    stride = 2

    model = ResNet1D(
        in_channels=n_chan_in, 
        base_filters=base_filters, 
        kernel_size=kernel_size, 
        stride=stride, 
        n_block=n_block, 
        groups=1,
        n_length=n_length, 
        downsample_gap=downsample_gap, 
        increasefilter_gap=increasefilter_gap, 
        verbose=False).to(device)

    s = summary(model,input_data=input_data.mT)
    print()
    print(model._get_name())
    print(s.input_size)
    print(s)


    # tcn
    dropout = 0.5
    kernel_size = 3
    filters_per_layer = [128,64, 32, 16, 32, 64, 128, 128]

    model = TCN1D(n_length, n_length, filters_per_layer, kernel_size, dropout).to(device)

    s = summary(model,input_data=input_data)
    print()
    print(model._get_name())
    print(s.input_size)
    print(s)


    print('Tests completed')