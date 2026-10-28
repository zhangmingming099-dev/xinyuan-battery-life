"""Unmodified function ASTs extracted from the existing training implementation."""
import math
import numpy as np

def transform(X, observed, prep, bins):
    if X.shape[-1] % bins:
        raise ValueError('charge-bins必须整除容量采样点数')
    med = np.array(prep['median'], np.float32)[None, :, None, None]
    lo = np.array(prep['min'], np.float32)[None, :, None, None]
    scale = np.array(prep['scale'], np.float32)[None, :, None, None]
    z = (np.where(np.isfinite(X), X, med) - lo) / scale
    b, c, n, p = z.shape
    z = z.reshape(b, c, n, bins, p // bins).mean(-1).transpose(0, 1, 3, 2)
    quality = observed.reshape(b, n, bins, p // bins).mean(-1).transpose(0, 2, 1)[:, None]
    return np.concatenate([z, quality], axis=1).astype(np.float32)

def build_network(torch, attention, bins):
    nn = torch.nn

    class Causal(nn.Module):

        def __init__(self, cin, cout, d):
            super().__init__()
            self.pad = 7 * d
            self.conv = nn.utils.parametrizations.weight_norm(nn.Conv2d(cin, cout, (1, 8), dilation=(1, d)))

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (self.pad, 0, 0, 0)))

    class ECA(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, 5, padding=2, bias=False)

        def forward(self, x):
            score = x.mean((2, 3)).unsqueeze(1)
            weights = torch.sigmoid(self.conv(score)).squeeze(1)[:, :, None, None]
            return x * weights

    class Block(nn.Module):

        def __init__(self, cin, cout, d):
            super().__init__()
            self.branch = nn.Sequential(Causal(cin, cout, d), nn.LeakyReLU(0.1), nn.Dropout(0.1), Causal(cout, cout, d), nn.LeakyReLU(0.1), nn.Dropout(0.1))
            self.attention = ECA() if attention else nn.Identity()
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
            self.act = nn.LeakyReLU(0.1)

        def forward(self, x):
            return self.act(self.attention(self.branch(x)) + self.skip(x))

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.fusion = nn.Conv2d(4, 1, 1)
            self.blocks = nn.Sequential(Block(1, 16, 1), Block(16, 32, 2), Block(32, 64, 5))
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            z = self.blocks(self.fusion(x))
            return self.head(z[:, :, :, -1].mean(2)).squeeze(1)
    return Net()

def predict(torch, net, values, indices, scaling, device, batch_size):
    net.eval()
    result = []
    with torch.no_grad():
        for batch in np.array_split(indices, math.ceil(len(indices) / batch_size)):
            v = net(torch.from_numpy(values[batch]).to(device)).cpu().numpy()
            result.extend((v * scaling['target_std'] + scaling['target_mean']).tolist())
    result = np.asarray(result)
    if len(result) != len(indices) or not np.isfinite(result).all():
        raise RuntimeError('预测长度异常或出现非有限值，停止而不生成虚假指标')
    return result
