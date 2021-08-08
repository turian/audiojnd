import torchopenl3
import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value] * 6144))

    def forward(self, input):
        return input * self.scale


class AudioJNDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchopenl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=6144
        )
        self.scale = ScaleLayer()
        self.cos = nn.CosineSimilarity(eps=1e-6)

    def forward(self, x1, x2):
        bs, in1, in2, in3 = x1.size()
        x1 = self.model(x1.view(-1, in2, in3))
        x2 = self.model(x2.view(-1, in2, in3))
        x1 = self.scale(x1).view(bs, -1)
        x2 = self.scale(x2).view(bs, -1)
        prob = self.cos(x1, x2)
        assert torch.all((prob < 1) & (prob > -1))
        prob = (prob + 1) / 2
        assert torch.all((prob < 1) & (prob > 0))
        return prob


if __name__ == "__main__":
    input1 = torch.randn(2, 17, 1, 48000)  # 2 is here batch_size
    input2 = torch.randn(2, 17, 1, 48000)

    model = AudioJNDModel()
    with torch.no_grad():
        prob = model(input1, input2)

    print(prob)
