![Normalization](https://pics.spark-in.me/upload/7c12fea58ff515ffb46df52b6050ace0.jpg)

# Russian STT Text Normalization

Russian text normalization pipeline for speech-to-text and other applications based on tagging s2s networks.

## Requirements

- Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4 for s2s pipeline
- [tqdm](https://github.com/tqdm/tqdm) for progress bar

```
pip install torch
pip install tqdm
```

## Usage

```python
from normalizer import Normalizer

text = 'С 12.01.1943 г. площадь сельсовета — 1785,5 га.'

norm = Normalizer()
result = norm.norm_text(text)
print(result)
```

```
>>> С двенадцатого января тысяча девятьсот сорок третьего года площадь сельсовета
>>> — тысяча семьсот восемьдесят пять целых и пять десятых гектара
```
