from mmseg.models.decode_heads.segmenter_mask_head import SegmenterMaskTransformerHead
import torch
import torch.nn.functional as F

class SegmenterMaskTransformerHeadv2(SegmenterMaskTransformerHead):
    def __init__(self, num_classes= 1, in_channels=[1536,1536,1536,1536], channels=6144, image_shape=224, **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            in_index=(0, 1, 2, 3),
            num_layers=12,
            num_heads=12,
            embed_dims=768,
            input_transform="resize_concat",
            num_classes=num_classes,
            **kwargs
        )
        self.image_shape = image_shape

    def forward(self, inputs, depth=None):
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
        
        x = self._transform_inputs(inputs)
        # print(f'x shape: {x.shape}')
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        # NOTE: add upsample
        masks = torch.nn.functional.interpolate(masks, size=self.image_shape, mode="bilinear", align_corners=self.align_corners)

        return masks

        

if __name__ == "__main__":
    segmenter = SegmenterMaskTransformerHeadv2()
    multi_level_feat = [(torch.rand(1, 768, 16, 16), torch.rand(1, 768)), (torch.rand(1, 768, 16, 16), torch.rand(1, 768)), \
                        (torch.rand(1, 768, 16, 16), torch.rand(1, 768)), (torch.rand(1,768, 16, 16), torch.rand(1, 768))]
    
    output = segmenter(multi_level_feat)

    print(output.shape)

