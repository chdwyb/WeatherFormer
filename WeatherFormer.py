import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from base_modules import OverlapPatchEmbed, LayerNorm, Downsample, Upsample


# perspective-unified integrated attention (PIA)
# i. dual sparse permuted self-attention (DSPA)
# ii. vision-broad convolution attention (VCA)
class PIA(nn.Module):
    def __init__(self, dim, n_div=1, num_heads=8, bias=False, task_query=False, ssr=4, top_k=True):
        super(PIA, self).__init__()
        self.num_heads = num_heads
        self.task_query = task_query
        self.ssr = ssr
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.n_div = n_div
        if self.n_div > 1:
            self.qkv_dim = dim // n_div
            self.s_dim = dim - self.qkv_dim
            if self.task_query:
                self.query = nn.Parameter(torch.ones(self.qkv_dim, 48))

                self.q = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=1, bias=bias)
                self.q_dwconv = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=3, stride=1, padding=1,
                                            groups=self.qkv_dim, bias=bias)
                self.kv = nn.Conv2d(self.qkv_dim, self.qkv_dim * 2, kernel_size=1, bias=bias)
                self.kv_dwconv = nn.Conv2d(self.qkv_dim * 2, self.qkv_dim * 2, kernel_size=3, stride=1, padding=1,
                                            groups=self.qkv_dim * 2, bias=bias)
            else:
                if self.ssr > 1:
                    self.ss = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=self.ssr, stride=self.ssr)
                    self.norm = LayerNorm(self.qkv_dim)

                    self.qk = nn.Conv2d(self.qkv_dim, self.qkv_dim * 2, kernel_size=1, bias=bias)
                    self.qk_dwconv = nn.Conv2d(self.qkv_dim * 2, self.qkv_dim * 2, kernel_size=3, stride=1, padding=1,
                                                groups=self.qkv_dim * 2, bias=bias)

                    self.v = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=1, bias=bias)
                    self.v_dwconv = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=3, stride=1, padding=1,
                                                groups=self.qkv_dim, bias=bias)
                else:
                    self.qkv = nn.Conv2d(self.qkv_dim, self.qkv_dim * 3, kernel_size=1, bias=bias)
                    self.qkv_dwconv = nn.Conv2d(self.qkv_dim * 3, self.qkv_dim * 3, kernel_size=3, stride=1, padding=1,
                                               groups=self.qkv_dim * 3, bias=bias)
            self.a = nn.Sequential(
                nn.Conv2d(self.s_dim, self.s_dim, 1, bias=bias),
                nn.GELU(),
                nn.Conv2d(self.s_dim, self.s_dim, 11, padding=5, groups=self.s_dim, bias=bias)
            )
            self.s = nn.Conv2d(self.s_dim, self.s_dim, 1, bias=bias)
            if self.top_k:
                self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        elif self.n_div == 1:
            self.qkv_dim = dim
            if self.task_query:
                self.query = nn.Parameter(torch.ones(self.qkv_dim, 16, 16))

                self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
                self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
            else:
                if self.ssr > 1:
                    self.ss = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=self.ssr, stride=self.ssr)
                    self.norm = LayerNorm(self.qkv_dim)

                    self.qk = nn.Conv2d(self.qkv_dim, self.qkv_dim * 2, kernel_size=1, bias=bias)
                    self.qk_dwconv = nn.Conv2d(self.qkv_dim * 2, self.qkv_dim * 2, kernel_size=3, stride=1, padding=1,
                                                groups=self.qkv_dim * 2, bias=bias)

                    self.v = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=1, bias=bias)
                    self.v_dwconv = nn.Conv2d(self.qkv_dim, self.qkv_dim, kernel_size=3, stride=1, padding=1,
                                                groups=self.qkv_dim, bias=bias)
                else:
                    self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
                    self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
            if self.top_k:
                self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
                self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        elif self.n_div == 0:
            self.s_dim = dim
            self.a = nn.Sequential(
                nn.Conv2d(self.s_dim, self.s_dim, 1, bias=bias),
                nn.GELU(),
                nn.Conv2d(self.s_dim, self.s_dim, 11, padding=5, groups=self.s_dim, bias=bias)
            )
            self.s = nn.Conv2d(self.s_dim, self.s_dim, 1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        if self.n_div > 1:
            x1, x2 = torch.split(x, [self.qkv_dim, self.s_dim], dim=1)
            b, c, h, w = x1.shape
            if self.task_query:
                q = self.query.unsqueeze(0).repeat(b, 1, 1, 1)
                q = self.q_dwconv(self.q(q))
                q = F.interpolate(q, size=(h, w))

                kv = self.kv_dwconv(self.kv(x1))
                k, v = kv.chunk(2, dim=1)
            else:
                if self.ssr > 1:
                    qk = self.norm(self.ss(x1))
                    _, _, h_qk, w_qk = qk.shape
                    qk = self.qk_dwconv(self.qk(qk))
                    q, k = qk.chunk(2, dim=1)
                    v = self.v_dwconv(self.v(x1))

                    q = rearrange(q, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
                    k = rearrange(k, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
                    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                else:
                    qkv = self.qkv_dwconv(self.qkv(x1))
                    q, k, v = qkv.chunk(3, dim=1)

                    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            if self.top_k:
                _, _, C, _ = q.shape

                mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

                attn = (q @ k.transpose(-2, -1)) * self.temperature

                index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
                mask1.scatter_(-1, index, 1.)
                attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
                mask2.scatter_(-1, index, 1.)
                attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
                mask3.scatter_(-1, index, 1.)
                attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
                mask4.scatter_(-1, index, 1.)
                attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

                attn1 = attn1.softmax(dim=-1)
                attn2 = attn2.softmax(dim=-1)
                attn3 = attn3.softmax(dim=-1)
                attn4 = attn4.softmax(dim=-1)

                out1 = (attn1 @ v)
                out2 = (attn2 @ v)
                out3 = (attn3 @ v)
                out4 = (attn4 @ v)

                x1 = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
            else:
                attn = (q @ k.transpose(-2, -1)) * self.temperature
                attn = attn.softmax(dim=-1)
                x1 = (attn @ v)
            x1 = rearrange(x1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            x2 = self.a(x2) * self.s(x2)
            x = torch.cat((x1, x2), 1)
        elif self.n_div == 1:
            b, c, h, w = x.shape
            if self.task_query:
                q = self.query.unsqueeze(0).repeat(b, 1, 1, 1)
                q = self.q_dwconv(self.q(q))
                q = F.interpolate(q, size=(h, w))

                kv = self.kv_dwconv(self.kv(x))
                k, v = kv.chunk(2, dim=1)
            else:
                if self.ssr > 1:
                    qk = self.norm(self.ss(x))
                    _, _, h_qk, w_qk = qk.shape
                    qk = self.qk_dwconv(self.qk(qk))
                    q, k = qk.chunk(2, dim=1)
                    v = self.v_dwconv(self.v(x))

                    q = rearrange(q, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
                    k = rearrange(k, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
                    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                else:
                    qkv = self.qkv_dwconv(self.qkv(x))
                    q, k, v = qkv.chunk(3, dim=1)

                    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            if self.top_k:
                _, _, C, _ = q.shape

                mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
                mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

                attn = (q @ k.transpose(-2, -1)) * self.temperature

                index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
                mask1.scatter_(-1, index, 1.)
                attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
                mask2.scatter_(-1, index, 1.)
                attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
                mask3.scatter_(-1, index, 1.)
                attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

                index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
                mask4.scatter_(-1, index, 1.)
                attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

                attn1 = attn1.softmax(dim=-1)
                attn2 = attn2.softmax(dim=-1)
                attn3 = attn3.softmax(dim=-1)
                attn4 = attn4.softmax(dim=-1)

                out1 = (attn1 @ v)
                out2 = (attn2 @ v)
                out3 = (attn3 @ v)
                out4 = (attn4 @ v)

                x = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
            else:
                attn = (q @ k.transpose(-2, -1)) * self.temperature
                attn = attn.softmax(dim=-1)
                x = (attn @ v)
            x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        elif self.n_div == 0:
            x = self.a(x) * self.s(x)
        x = self.project_out(x)
        return x


# progressive gated feed-forward network (PGFN)
class PGFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2, bias=False):
        super(PGFN, self).__init__()
        hidden_features = int(dim * mlp_ratio)
        split_dim = int(hidden_features // 4)
        self.split_dim = split_dim

        self.gconv_1 = nn.Conv2d(split_dim, split_dim, kernel_size=1, groups=split_dim, bias=bias)
        self.gconv_2 = nn.Conv2d(split_dim, split_dim, kernel_size=3, stride=1, padding=1, groups=split_dim, bias=bias)
        self.gconv_3 = nn.Conv2d(split_dim, split_dim, kernel_size=7, stride=1, padding=3, groups=split_dim, bias=bias)
        self.gconv_4 = nn.Conv2d(split_dim, split_dim, kernel_size=11, stride=1, padding=5, groups=split_dim, bias=bias)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)

        x1_1, x1_2, x1_3, x1_4 = x1.chunk(4, dim=1)
        x1_1 = self.gconv_1(x1_1)
        x1_2 = self.gconv_2(x1_2)
        x1_3 = self.gconv_3(x1_3)
        x1_4 = self.gconv_4(x1_4)

        # # progressive gating mechanism
        # x1_3 = x1_3 * x1_4
        # x1_2 = x1_2 * x1_3
        # x1_1 = x1_1 * x1_2

        x1 = torch.cat((x1_1, x1_2, x1_3, x1_4), 1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_div=1, num_heads=8, mlp_ratio=2., bias=False, ssr=4, task_query=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = PIA(dim, n_div=n_div, num_heads=num_heads, ssr=ssr, task_query=task_query)
        self.norm2 = LayerNorm(dim)
        self.ffn = PGFN(dim, mlp_ratio=mlp_ratio, bias=bias)

        # layer scale
        layer_scale_init_value = 0.
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x))
        return x


# WeatherFormer
class WeatherFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 n_div=(1, 2, 4, 8),  # partition coefficient
                 num_blocks=(4, 6, 6, 8),
                 ssrs=(4, 2, 2, 1),  # spatial sparse ratios
                 heads=(1, 2, 4, 8),
                 mlp_ratios=2
                 ):
        super(WeatherFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, n_div=n_div[0], num_heads=heads[0], mlp_ratio=mlp_ratios, ssr=ssrs[0]) for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), n_div=n_div[1], num_heads=heads[1], mlp_ratio=mlp_ratios, ssr=ssrs[1]) for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), n_div=n_div[2], num_heads=heads[2], mlp_ratio=mlp_ratios, ssr=ssrs[2]) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), n_div=n_div[3], num_heads=heads[3], mlp_ratio=mlp_ratios, ssr=ssrs[3]) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), n_div=n_div[2], num_heads=heads[2], mlp_ratio=mlp_ratios, ssr=ssrs[2],) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), n_div=n_div[1], num_heads=heads[1], mlp_ratio=mlp_ratios, ssr=ssrs[1],) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), n_div=n_div[0], num_heads=heads[0], mlp_ratio=mlp_ratios, ssr=ssrs[0],) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), n_div=n_div[0], num_heads=heads[0], mlp_ratio=mlp_ratios, ssr=ssrs[3],) for _ in range(4)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # encoder 1
        input_ = x  # global connection
        x1 = self.patch_embed(x)
        x1 = self.encoder_level1(x1)
        x1_skip = x1  # skip connection

        # encoder 2
        x2 = self.down1_2(x1)
        x2 = self.encoder_level2(x2)
        x2_skip = x2  # skip connection

        # encoder 3
        x3 = self.down2_3(x2)
        x3 = self.encoder_level3(x3)
        x3_skip = x3  # skip connection

        # encoder 4
        x4 = self.down3_4(x3)
        x4 = self.latent(x4)

        # decoder 4
        x3 = self.up4_3(x4)
        x3 = torch.cat([x3, x3_skip], 1)
        x3 = self.reduce_chan_level3(x3)
        x3 = self.decoder_level3(x3)

        # decoder 3
        x2 = self.up3_2(x3)
        x2 = torch.cat([x2, x2_skip], 1)
        x2 = self.reduce_chan_level2(x2)
        x2 = self.decoder_level2(x2)

        # decoder 2
        x1 = self.up2_1(x2)
        x1 = torch.cat([x1, x1_skip], 1)
        x1 = self.decoder_level1(x1)

        # decoder 1
        x = self.refinement(x1)
        x = self.output(x) + input_
        return x, x4


if __name__ == '__main__':
    torch.cuda.empty_cache()
    x = torch.randn((1, 3, 224, 224)).cuda()
    net = WeatherFormer().cuda()
    y = net(x)
    print(y.shape)

    # import time
    # start = time.time()
    # for i in range(100):
    #     y = net(x)
    # end = time.time()
    # print(f'{(end - start) / 100}')


