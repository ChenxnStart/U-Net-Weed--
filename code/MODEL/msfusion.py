import torch.nn as nn
# 确保这个引用路径对您的项目结构是正确的
from code.MODEL.model import MSFusionUNet as CoreNet

class MSFusion(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, norm_type='bn', dilation=2, **kwargs):
        """
        参数说明：
        in_channels: 框架自动根据 YAML 的 channels 列表长度传入
        num_classes: 框架自动根据 dataset 配置传入
        norm_type: YAML 中的 params.norm_type
        dilation: YAML 中的 params.dilation
        """
        super().__init__()
        self.model = CoreNet(
            in_channels=in_channels,
            num_classes=num_classes,
            norm_type=norm_type,
            dilation=dilation
        )

    def forward(self, x):
        return self.model(x)

# 某些框架需要入口函数
def get_model(**kwargs):
    return MSFusion(**kwargs)
