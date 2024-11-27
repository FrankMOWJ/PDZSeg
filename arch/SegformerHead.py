import torch
from torch import nn
from mmseg.models.decode_heads import SegformerHead
from mmseg.models.utils import resize
from mmcv.cnn import ConvModule

class SegformerHeadv2(SegformerHead):
    def __init__(self, 
        interpolate_mode='bilinear', 
        depth_channel=1536,
        num_classes=3,
        in_channels=[1536, 1536, 1536, 1536], # base
        channels=384,
        image_shape=(224, 224),
        **kwargs):

        # 移除 kwargs 中的 in_channels，避免传递多个值
        kwargs.pop('in_channels', None) 

        super().__init__(interpolate_mode, 
                        in_channels=in_channels,
                        in_index=(0, 1, 2, 3),
                        channels=channels, # self.convs out_channel
                        num_classes=num_classes,
                        **kwargs)
        self.fusion_conv = ConvModule(
            in_channels = self.channels * len(self.in_index) + depth_channel,
            out_channels = self.channels,
            kernel_size = 1,
            norm_cfg = self.norm_cfg
        )
        self.img_resolution = image_shape
        self.upsample_rate = 4

    def forward(self, inputs, depth_maps=None):
        
        # Receive 4 stage backbone feature map
        # inputs = self._transform_inputs(inputs)
        inputs = list(inputs)

        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        # for i, feature in enumerate(inputs):
            # inputs[i] = torch.nn.functional.interpolate(feature, size=self.img_resolution, mode="bilinear", align_corners=self.align_corners)
        # print(f'inchannels: {self.in_channels}')
        # print(f'image shape: {self.img_resolution}')
        # for i in range(len(inputs)):
        #     print(f'feature {i} shape: {inputs[i][0].shape}, {inputs[i][1].shape}')

        outs = []
            
        for idx in range(len(inputs)):
            x = inputs[idx]

            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size= [s * self.upsample_rate for s in inputs[0].shape[2:]],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = torch.cat(outs, dim=1)

        # out shape torch.Size([1, self.channel * 4(1536), 224, 224]), depth shape: torch.Size([1, depth_channel(768/1536), 224, 224])
        # print(f"out shape {outs.shape}")
        if depth_maps is not None:
            if len(depth_maps.shape) != 4:
                depth_maps = depth_maps.unsqueeze(dim=1)
            outs = torch.cat((outs, depth_maps), dim=1)
        # print(f"x shape {outs.shape}")
        out = self.fusion_conv(outs)

        out = self.cls_seg(out)

        # print(f'out shape: {out.shape}')
        out = torch.nn.functional.interpolate(out, size=self.img_resolution, mode="bilinear", align_corners=self.align_corners)

        return out
    # def forward(self, inputs):
    #     # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
    #     inputs = self._transform_inputs(inputs)
    #     outs = []
    #     for idx in range(len(inputs)):
    #         x = inputs[idx]
    #         conv = self.convs[idx]
    #         outs.append(
    #             resize(
    #                 input=conv(x),
    #                 size=self.img_resolution, # inputs[0].shape[2:],
    #                 mode=self.interpolate_mode,
    #                 align_corners=self.align_corners))

    #     out = self.fusion_conv(torch.cat(outs, dim=1))

    #     out = self.cls_seg(out)

    #     return out 

if __name__ == "__main__":
    segformerHead = SegformerHeadv2()
    multi_level_feat = [(torch.rand(1, 768, 224, 224), torch.rand(1, 768)), (torch.rand(1, 768, 112, 112), torch.rand(1, 768)), \
                        (torch.rand(1, 768, 112, 112), torch.rand(1, 768)), (torch.rand(1,768, 54, 54), torch.rand(1, 768))]
    
    output = segformerHead(multi_level_feat, torch.rand(1, 1536, 224, 224))

    print(output.shape)