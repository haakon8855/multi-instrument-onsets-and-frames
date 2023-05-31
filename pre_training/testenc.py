"""haakoas, matsjno"""

import sys
from functools import reduce
from torch.nn.modules.module import _addindent

import torch
from torch import nn
import torchvision

# from conv_stack import ConvStack
# from lstm import BiLSTM


class Encoder(nn.Module):
    """
    Class contaning an encoder for the Simple Siamese network.
    """

    def __init__(self):
        super().__init__()
        input_features = 229
        model_size = 229 * 3
        output_size = 640 * 5
        self.network = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet18()
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.BatchNorm1d(768),
        )

    def forward(self, x):
        x = x[:, None, :, :]  # Add empty dimension to act as number of channels
        x = self.network(x)  # Expand number of channels to 3
        x = self.resnet(x)
        # x = self.projector(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count


def main():
    """
    Main function for running this python script.
    """
    # device = torch.device("cuda")
    encoder = Encoder()  # .to(device)
    summary(encoder)
    print(encoder(torch.rand(8, 640, 229)).shape)


if __name__ == "__main__":
    main()
