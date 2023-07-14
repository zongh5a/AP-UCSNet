## Uncertainty awareness with adaptive propagation for multi-view stereo


## Training

1. Prepare DTU training set(640x512).
1. Edit config.py: set "DatasetsArgs.root_dir", "LoadDTU.train_root&train_pair".
2. Run the script for training.

```
python train.py  
```

## Testing

The pre-training model in "pth". 

1. Prepare DTU test set(1600x1200)([百度网盘](https://pan.baidu.com/s/15hZZ3eY2bSZnae5M079gIQ 
) 提取码：6au3) and Tanks and Temples dataset([百度网盘](https://pan.baidu.com/s/1pAcLFXwi_FGxQUM47JjIMw 
   ) 提取码：a4oz).
2. Edit config.py: set "DatasetsArgs.root_dir", "LoadDTU.eval_root&eval_pair", and "LoadTanks.eval_root"
3. Run the script for the test.

```
# DTU
python eval.py -p pth/dtu_16.pth -d dtu
# Tanks and Temples
python eval.py -p pth/dtu_16.pth -d tanks
```


## Fusion

There two methods in "tools": "filter"and "gipuma".

### DTU dataset 

1. Install fusibile tools: https://github.com/kysucix/fusibile
2. Edit tools/gipuma/conf.py: set "root_dir", "eval_folder" and "fusibile_exe_path".
3. Run the script.

```
cd tools/gipuma
python fusion.py -cfmgd
```

### Tanks and Temples dataset

1. Run the script.

```
# filter
cd tools/filter
python dynamic_filter_gpu.py -e EVAL_OUTPUT_LOCATION -r DATASET_PATH -o OUTPUT_PATH 
```


## Acknowledgements

Our work is partially baed on these opening source work: [MVSNet](https://github.com/YoYo000/MVSNet), [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch), [D2HC-RMVSNet](https://github.com/yhw-yhw/D2HC-RMVSNet).
We appreciate their contributions to the MVS community.


## Citation

This work will be published in _Applied Intelligence_.


<!-- If you find our code or paper helps, please cite:

```
@
```
-->