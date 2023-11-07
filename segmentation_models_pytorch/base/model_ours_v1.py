import torch
import torch.nn as nn
from . import initialization as init
import torch.nn.functional as F

from .lightrfb import LightRFB

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        Gx = torch.norm(inputs, p=2, dim=(2, 3), keepdim=True) # [1, 1, 1, c]
        x = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x
    
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class FAM(nn.Module):
    def __init__(self, in_channels):
        super(FAM, self).__init__()
        self.se_block1 = SEBlock(in_channels//3 * 4, in_channels//3*2)
        self.se_block2 = SEBlock(in_channels//3 * 4, in_channels//3*2)

        self.reduce = nn.Conv2d(in_channels//3*2, in_channels//3, 3, 1, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels//3)
    
    def shift(self, x, n_segment=3, fold_div=8):
        z = torch.chunk(x, 3, dim=1)
        x = torch.stack(z,dim=1)
        b, nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(b, n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :, :-1, :fold] = x[:, :, 1:, :fold]  # shift left
        out[:, :, 1:, fold: 2 * fold] = x[:, :, :-1, fold: 2 * fold]  # shift right
        out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift

        return out.view(b, nt*c, h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        # temporal shift
        x = self.shift(x)
        # import pdb; pdb.set_trace()
        y = torch.fft.fft2(x)
        y_imag = y.imag
        y_real = y.real

        y1_imag, y2_imag, y3_imag = torch.chunk(y_imag, 3, dim=1)
        y1_real, y2_real, y3_real = torch.chunk(y_real, 3, dim=1)

        # grouping
        pair1 = torch.concat([y1_imag, y2_imag, y1_real, y2_real], dim=1)
        pair2 = torch.concat([y1_imag, y3_imag, y1_real, y3_real], dim=1)

        pair1 = self.se_block1(pair1).float()
        pair2 = self.se_block2(pair2).float()

        y1_real, y1_imag = torch.chunk(pair1, 2, dim=1)
        y1 = torch.complex(y1_real, y1_imag)
        z1 = torch.fft.ifft2(y1, s=(h, w)).float()

        y2_real, y2_imag = torch.chunk(pair2, 2, dim=1)
        y2 = torch.complex(y2_real, y2_imag)
        z2 = torch.fft.ifft2(y2, s=(h, w)).float()
        
        out = self.reduce(z1 + z2)
        out = F.relu(out)
        out = self.norm(out)
        
        return out

class HeatmapHead(nn.Module):
    
    def __init__(self, input_channels, internal_neurons, out_channels):
        super(HeatmapHead, self).__init__()

        self.upsample1 = nn.ConvTranspose2d(input_channels, internal_neurons, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(internal_neurons, out_channels, 3, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(out_channels, 1, 3, stride=2, padding=1)

    def forward(self, inputs):
        outs = []
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(0)
        w, h = inputs.shape[-2:]
        x = self.upsample1(inputs, output_size=[w*2, h*2])
        outs.append(x)
        x = self.upsample2(x, output_size=[w*4, h*4])
        outs.append(x)
        x = self.upsample3(x, output_size=[w*8, h*8])
        outs.append(x)

        return outs

class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

        FAM_list = []
        heatmaphead_list = []

        in_channels = [1024]
        for in_c in in_channels:
            FAM_list.append(FAM(3 * in_c))
            heatmaphead_list.append(HeatmapHead(in_c, 256, 64))

        self.FAM_list = nn.ModuleList(FAM_list)
        self.heatmaphead_list = nn.ModuleList(heatmaphead_list)


    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)
        fam_outputs = []
        if x.ndim == 5:
            b, f, _, _, _ = x.shape
            x = x.flatten(0, 1)
            feats = self.encoder(x) # -> 6 diff scales of feats
            features = []
            pick_idxs = [4]
            for idx, fea in enumerate(feats):
                c, w, h = fea.shape[1:]
                fea = fea.view(b, f, c, w, h)
                if idx in pick_idxs:
                    pick_idx = idx - pick_idxs[0]
                    tmp_list, tmp_feat_list = [], []
                    for i in range(b):
                        curr_clip_feats = torch.cat([
                            fea[i][0].unsqueeze(0), 
                            fea[i][1].unsqueeze(0), 
                            fea[i][2].unsqueeze(0)
                            ], dim=1)
                        tmp_feat = self.FAM_list[pick_idx](curr_clip_feats)
                        y = fea[i][0] + tmp_feat.squeeze()
                        tmp_feat_list.append(tmp_feat)
                        tmp_list.append(y)

                    features.append(torch.stack(tmp_list))
                    fam_outputs.append(torch.stack(tmp_feat_list))

                else:
                    features.append(fea[:,0,:,:])
                
        else:
            features = self.encoder(x)

        heatmap_decoder = []
        for idx, feat in enumerate(fam_outputs):
            heatmap_decoder.append(self.heatmaphead_list[idx](feat.squeeze()))

        decoder_output = self.decoder(*features)

        # localization branch
        heatmap_pred1 = F.interpolate(heatmap_decoder[0][-1], size=features[0].shape[-2:], mode="bilinear")
        decoder_output = (decoder_output + heatmap_pred1)
        
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, [heatmap_pred1]
        # return masks


    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x