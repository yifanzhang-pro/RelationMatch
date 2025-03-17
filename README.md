# RelationMatch


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/relationmatch-matching-in-batch-relationships/semi-supervised-image-classification-on-stl-3)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-stl-3?p=relationmatch-matching-in-batch-relationships)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.10397)
![Python 3.10](https://img.shields.io/badge/python-3.7-green.svg)

Official implementation of paper "RelationMatch: Matching In-batch Relationships for Semi-supervised Learning" (https://arxiv.org/abs/2305.10397).

- **Achieving 15.21% accuracy improvement over FlexMatch on the STL-10 dataset!** 



## Installation

You can create the `ssl` environment by just running this command: 
`conda env create -f environment.yaml`

Then you may activate `ssl` environment by `conda activate ssl`

## Runnning experiments

Under `ssl` environment, run following command:

`python BASE_METHOD.py --c=PATH/TO/YOUR/CONFIG`

`BASE_METHOD` can be one of fixmatch, flexmatch, with respect to RelationMatch and RelationMatch(w/ CPL) in paper.

We provide our example config in `configs/BASE_METHOD/` folder with `relation` suffix. You can use `diff` to find the difference between relation match configs and TorchSSL configs, and it's easy to modify other TorchSSL configs to RelationMatch configs follow our sample configs. 


## Acknowledgement

We implement our RelationMatch method based on [TorchSSL](https://github.com/TorchSSL/TorchSSL) repo. Thanks for their wonderful work!


## Citations
Please cite the paper and star this repo if you use RelationMatch and find it interesting/useful, thanks! 

```bibtex
@article{zhang2023relationmatch,
  title={RelationMatch: Matching In-batch Relationships for Semi-supervised Learning},
  author={Zhang, Yifan and Yang, Jingqin and Tan, Zhiquan and Yuan, Yang},
  journal={arXiv preprint arXiv:2305.10397},
  year={2023}
}
```



