import lava.lib.dl.slayer as slayer
import lava.lib.dl.bootstrap as bootstrap
import torch
import h5py
from params import TIMESTEPS
from utils import LatencyEncoder


IEM_WEIGHT = 50
WEIGHT = 10
IEM_AMP = 3

class IEM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, neuron_params,weight_amp):
        super(IEM, self).__init__()

        ## Sequential Path (combines local + global concepts) ##
        self.layers = torch.nn.ModuleList([
            # Expand channels (from local path concept)
            bootstrap.block.cuba.Conv(neuron_params, in_features=in_channels,
                                     out_features=hidden_channels,
                                     kernel_size=1, stride=1, padding=0,
                                     weight_norm=False, weight_scale=weight_amp),
            # Depthwise spatial (from local path)
            bootstrap.block.cuba.Conv(neuron_params, in_features=hidden_channels,
                                     out_features=hidden_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     weight_norm=False, weight_scale=weight_amp,
                                     groups=hidden_channels),
            # Horizontal directional convolution (from global path)
            bootstrap.block.cuba.Conv(neuron_params, in_features=hidden_channels,
                                     out_features=hidden_channels,
                                    #  groups=hidden_channels,
                                     kernel_size=(1, 5), stride=1, padding=(0, 2),
                                     weight_norm=False, weight_scale=weight_amp),
            # Vertical directional convolution (from global path)
            bootstrap.block.cuba.Conv(neuron_params, in_features=hidden_channels,
                                     out_features=hidden_channels,
                                    #  groups=hidden_channels,
                                     kernel_size=(5, 1), stride=1, padding=(2, 0),
                                     weight_norm=False, weight_scale=weight_amp),
            # Compress to output channels (from local path)
            bootstrap.block.cuba.Conv(neuron_params, in_features=hidden_channels,
                                     out_features=out_channels,
                                     kernel_size=1, stride=1, padding=0,
                                     weight_norm=False, weight_scale=weight_amp),
        ])

    def forward(self, x, mode_list):
        # Sequential processing through all layers
        for i, block in enumerate(self.layers):
            x = block(x, mode=mode_list[i])
        return x

class ResidualIEM(torch.nn.Module):
    def __init__(self, iem: IEM, alpha: float = 1.0):
        super().__init__()
        self.iem = iem
        self.alpha = alpha

    def forward(self, x, mode_list_main, base_mode):
        y = self.iem(x, mode_list=mode_list_main)
        if base_mode == bootstrap.Mode.ANN:
            return x + self.alpha * y
        
        return torch.clamp(x + self.alpha* y, 0, 1) 
    
class ConcatIEM(torch.nn.Module):
    def __init__(self, iem: IEM, neuron_params, in_channels, out_channels):
        super().__init__()
        self.iem = iem
        self.merge_layer = bootstrap.block.cuba.Conv(
            neuron_params,
            in_features=(in_channels+out_channels),
            out_features=out_channels,
            kernel_size=1,stride=1,padding=0,
            weight_norm=False, weight_scale=WEIGHT
        )

    def forward(self, x, mode_list_main, merge_mode):
        y = self.iem(x,mode_list=mode_list_main)
        combined = torch.cat([x,y],dim=1)
        return self.merge_layer(combined,mode=merge_mode)


class Network(torch.nn.Module):
    def __init__(self, time_steps=TIMESTEPS):  # Reduced timesteps for CPU testing
        super(Network, self).__init__()
        self.time_steps = time_steps

        self.latency_encoder = LatencyEncoder(
            timesteps=time_steps,
            threshold=0.1
        )

        # Adjusted neuron parameters to enable actual spiking
        # Threshold must be low enough to spike within first batch
        neuron_params = {
                'threshold'     : 0.05, # 3x lower 
                'current_decay' : 1,     
                'voltage_decay' : 0.002,  # 10x lower
                'tau_grad'      : 1,
                'scale_grad'    : 10,
                'persistent_state': False,
            }
        neuron_params_norm = {
                **neuron_params,
                # 'norm'    : slayer.neuron.norm.MeanOnlyBatchNorm,
            }

        ### LD-Net Architecture ###
        ## Stem Module ##
        # Input: 1×H×W → Output: 8×H×W (same spatial dimensions)
        self.stem = torch.nn.ModuleList([
            # bootstrap.block.cuba.Input(neuron_params, weight=1.0, bias=0.0),
            # Dilated conv for expanded receptive field (5×5 effective, replaces upsample)
            bootstrap.block.cuba.Conv(neuron_params, in_features=1, out_features=32,
                                      kernel_size=3, stride=1, padding=2, dilation=2,
                                      weight_norm=False, weight_scale=WEIGHT),
            # Regular conv to integrate spatial information (no downsampling needed)
            bootstrap.block.cuba.Conv(neuron_params, in_features=32, out_features=32,
                                      kernel_size=3, stride=1, padding=1,
                                      weight_norm=False, weight_scale=WEIGHT),
            # Depthwise 3×3 convolution (groups=32 for depthwise)
            bootstrap.block.cuba.Conv(neuron_params, in_features=32, out_features=32,
                                      kernel_size=3, stride=1, padding=1,
                                      weight_norm=False, weight_scale=WEIGHT, groups=32),
            # Pointwise 1×1 convolution to reduce to 8 channels
            bootstrap.block.cuba.Conv(neuron_params, in_features=32, out_features=16,
                                      kernel_size=1, stride=1, padding=0,
                                      weight_norm=False, weight_scale=WEIGHT),
        ])

        ## IEM Modules ##
        # According to Table 2 in paper: IEM1(8→32→16), IEM2(16→64→32), IEM3(32→128→16), IEM4(16→64→8)
        self.iems = torch.nn.ModuleList([
            IEM(in_channels=16, hidden_channels=64, out_channels=16, neuron_params=neuron_params, weight_amp=IEM_WEIGHT),
            # IEM(in_channels=32, hidden_channels=64, out_channels=32, neuron_params=neuron_params, weight_amp=IEM_WEIGHT),
            # IEM(in_channels=32, hidden_channels=64, out_channels=16, neuron_params=neuron_params, weight_amp=IEM_WEIGHT),
        ])

        # for iem in self.iems:
        #     for name, param in iem.named_parameters():
        #         if 'synapse.weight' in name and param.dim() >= 2:
        #             param.data *= IEM_AMP


        ## Head Module ##
        # Input: 8×H×W → Output: 2×H×W (binary classification: RFI vs non-RFI)
        self.head = bootstrap.block.cuba.Conv(neuron_params, in_features=16, out_features=2, #Should it not be 1x1???
                                             kernel_size=1, stride=1, padding=0,
                                             weight_norm=False, weight_scale=WEIGHT)

    def forward(self, x, mode):
        N, C, H, W = x.shape
        if mode.base_mode == bootstrap.Mode.ANN:
            x = x.reshape([N, C, H, W, 1])
        else:
            x = self.latency_encoder(x)

        # Convert mode to iterator for sequential access
        mode_iter = iter(mode)

        # DEBUG: Enable layer-by-layer spike monitoring
        debug = False  # Set to True to see where signal dies

        if debug:
            print(f"  [FWD] Input: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Process through Stem (4 conv layers)
        for i, block in enumerate(self.stem):
            x = block(x, mode=next(mode_iter))
            if debug:
                spikes = (x > 0).sum().item()
                print(f"  [FWD] Stem layer {i+1}: spikes={spikes}/{x.numel()} ({100*spikes/x.numel():.2f}%), mean={x.mean():.4f}")

        # Process through IEMs (each IEM has 5 layers: sequential combination of local + global)
        for iem_idx, iem in enumerate(self.iems):
            # Extract 5 modes for this IEM
            iem_modes = [next(mode_iter) for _ in range(5)]
            x = iem(x, mode_list=iem_modes)
            if debug:
                spikes = (x > 0).sum().item()
                print(f"  [FWD] IEM {iem_idx+1}: spikes={spikes}/{x.numel()} ({100*spikes/x.numel():.2f}%), mean={x.mean():.4f}")

        # Process through Head (1 layer)
        x = self.head(x, mode=next(mode_iter))
        if debug:
            spikes = (x > 0).sum().item()
            print(f"  [FWD] Head: spikes={spikes}/{x.numel()} ({100*spikes/x.numel():.2f}%), mean={x.mean():.4f}")

        return x

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        simulation = h.create_group('simulation')
        simulation['Ts'] = 1
        simulation['tSample'] = self.time_steps
        layer = h.create_group('layer')

        # Export Stem layers
        layer_idx = 0
        for b in self.stem:
            b.export_hdf5(layer.create_group(f'{layer_idx}'))
            layer_idx += 1

        # Export IEM layers (each IEM has 5 sequential layers)
        for iem in self.iems:
            for b in iem.layers:
                b.export_hdf5(layer.create_group(f'{layer_idx}'))
                layer_idx += 1

        # Export Head layer
        self.head.export_hdf5(layer.create_group(f'{layer_idx}'))

