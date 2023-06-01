______________________________________________________________________

<div align="center">

# AVLIT: Audio-Visual Lightweight ITerative model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<!-- <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> -->
<!-- <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a> -->
<!-- <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br> -->
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

> 🚀: The code will be released soon.

## Description
Official Pytorch Lightning implementation of "Audio-Visual Speech Separation in Noisy Environments with a Lightweight Iterative Model", accepted at INTERSPEECH 2023.

|            |            |
|:----------:|:----------:|
|        |         |
| ![AVLIT_Folded](docs/AVLIT_Folded.png) | ![AVLIT_Unfolded](docs/AVLIT_Unfolded.png) |
| (A) Folded view of AVLIT | (B) Unfolded view of AVLIT |

Audio-Visual Lightweight ITerative model (AVLIT) uses the [A-FRCNN](https://github.com/JusperLee/AFRCNN-For-Speech-Separation) as building block. 
AVLIT employs a homogeneous design with audio and video branches composed of A-FRCNN blocks used iteratively. The weights are shared for each modality, making the number of parameters constant. Please refer to the paper for details.

## Quick start

### Installation

Make sure to have ``pytorch`` with GPU support installed on your machine according to the [official installation guide](https://pytorch.org/get-started/locally/).

### Basic usage

Here is a minimal example of how to use AVLIT in plain Pytorch. The default parameters will produce the configuration for AVLIT-8, which is the best performing model in the paper.

```python
from avlit import AVLIT

# Instantiate the model
model = AVLIT(
    num_sources = 2,
    # Audio branch
    audio_num_blocks = 8,
    # Video branch
    video_num_blocks = 4,
    video_encoder_checkpoint = "path/to/ae.ckpt",
)
model.cuda()

# Training or inference logic here
# ...

```

### Advanced usage

For more control over the architecture, it is possible to provide values for more parameters as follows:

```python
from avlit import AVLIT

# Instantiate the model
model = AVLIT(
    num_sources = 2,
    # Audio branch
    kernel_size = 40,
    audio_hidden_channels = 512,
    audio_bottleneck_channels = 128,
    audio_num_blocks = 8,
    audio_states = 5,
    # Video branch
    video_hidden_channels = 128,
    video_bottleneck_channels = 128,
    video_num_blocks = 4,
    video_states = 5,
    video_encoder_checkpoint = "path/to/ae.ckpt",
    video_encoder_trainable = False,
    video_embedding_dim = 1024,
    # AV fusion
    fusion_operation = "sum",
    fusion_positions = [4],
)
model.cuda()

# Training or inference logic here
# ...

```

## Cite

If you use AVLIT in your research, please cite our paper:
```
@inproceedings{avlit,
  title={Audio-Visual Speech Separation in Noisy Environments with a Lightweight Iterative Model},
  author={Martel, Héctor and Richter, Julius and Li, Kai and Hu, Xiaolin and Gerkmann, Timo},
  booktitle={Interspeech}
  year={2023}
}
```

## Contact

* For **technical/academic questions** please write an email to the corresponding authors mentioned in the paper. Alternatively, use the [discussions](https://github.com/hmartelb/avlit/discussions) page. Do not open an issue.
* For **bugs** or **problems with the code**, please [open an issue](https://github.com/hmartelb/avlit/issues) in this repository.
* For **other inquiries**, contact me via email at hmartelb@hotmail.com. 

## License

This code is licensed under the terms of the MIT License.

```
MIT License
Copyright (c) 2023 Héctor Martel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
