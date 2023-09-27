# TorchSSL-RelationMatch

We implement our RelationMatch method based on [TorchSSL](https://github.com/TorchSSL/TorchSSL) repo. Thanks for their wonderful work!

## Prepare for environment

You can create the `ssl` environment by just running this command: 
`conda env create -f environment.yaml`

Then you may activate `ssl` environment by `conda activate ssl`

## Run paper experiments

Under `ssl` environment, run following command:

`python BASE_METHOD.py --c=PATH/TO/YOUR/CONFIG`

`BASE_METHOD` can be one of fixmatch, flexmatch, with respect to RelationMatch and RelationMatch(w/ CPL) in paper.

We provide our example config in `configs/BASE_METHOD/` folder with `relation` suffix. You can use `diff` to find the difference between relation match configs and TorchSSL configs, and it's easy to modify other TorchSSL configs to RelationMatch configs follow our sample configs. 





