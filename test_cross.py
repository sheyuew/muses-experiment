import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # x1 和 x2 的形状均为 [batch_size, channels, height, width]
        
        # 生成查询、键和值
        Q = self.query_conv(x1)  # [batch_size, channels, height, width]
        K = self.key_conv(x2)     # [batch_size, channels, height, width]
        V = self.value_conv(x2)   # [batch_size, channels, height, width]

        # 将特征图展平以进行注意力计算
        batch_size, channels, height, width = Q.size()
        
        Q_flat = Q.view(batch_size, channels, -1)   # [batch_size, channels, height * width]
        K_flat = K.view(batch_size, channels, -1)   # [batch_size, channels, height * width]
        V_flat = V.view(batch_size, channels, -1)   # [batch_size, channels, height * width]

        # 计算注意力权重
        attention_scores = torch.bmm(Q_flat.transpose(1, 2), K_flat) / (K_flat.size(-1) ** 0.5)  # [batch_size, height*width, height*width]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        context = torch.bmm(attention_weights, V_flat.transpose(1, 2))  # [batch_size, height*width, channels]
        
        # 恢复形状并返回
        context = context.view(batch_size, channels, height, width)      # [batch_size, channels, height, width]
        
        return context + x1  # 可以选择加法或其他融合方式


# 示例使用
if __name__ == "__main__":
    batch_size = 2
    in_channels = 96
    height = 64
    width = 64

    x1 = torch.rand(batch_size, in_channels, height, width)   # 输入特征图1
    x2 = torch.rand(batch_size, in_channels, height, width)   # 输入特征图2

    cross_attention_block = CrossAttentionBlock(in_channels)
    
    output = cross_attention_block(x1, x2)

    print("Output shape:", output.shape)  # 应该输出 [2，96，256，256]