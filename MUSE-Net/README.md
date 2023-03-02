# Disentangling Multi-Pattern for Traffic Flow Forecasting (MUSE-Net) 
#### *by: Jianyang Qin, Ruiqi Zhang, Ye Ding, Yan Jia, Xuan Wang and Qing Liao*



## Abstract
<p align="center">
<img src="./misc/framework.jpg" width="800" class="center">
</p>

Traffic flow forecasting plays a crucial role in building smart cities in the Mobile Internet age. However, existing works of traffic flow forecasting investigated the multi-periodicity (e.g., hourly, daily and weekly) of traffic flow in an entangled manner, leading to unsatisfactory prediction of traffic flow. This is because the entangled learning that learns an unified representation for multi-periodicity has not yet to deal with the distribution shift and dynamic temporal interaction problems. In this paper, we propose a novel disentangled learning framework, namely <b>MU</b>lti-Pattern Di<b>SE</b>ntanglement Network (<b>MUSE-Net</b>), to tackle the limitations of entangled learning. Grounded in the theory of mutual information, we first learn and disentangle exclusive and interactive representations of traffic flow from multi-periodic patterns. Then, we utilize semantic-pushing and semantic-pulling regularizations to encourage the learned representations to be independent and informative. Moreover, we derive a lower bound estimator to tractably optimize the disentanglement problem and propose a joint training model for traffic flow forecasting. Extensive experimental results on several real-world traffic datasets demonstrate the effectiveness of the proposed framework.


## Requirements:
- Python 3.6
- tensorflow-gpu ==1.3.0
- keras == 2.0.8
- Numpy
- Pandas
- h5py == 2.9.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1zeXvNfDu1BbDvgqcC7HupQ), password: tgoh. 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Train:
You can train our MUSE-Net by following the steps bellow.

 - Download the datasets and put them in "Data" folder

 - Run with "python train.py" for NYC-Bike dataset, or "python train.py --dataset BikeNYC --device 0" for NYC-Bike dataset using GPU device 0

   ```
   python train.py
   ```

   ```
   python train.py --dataset BikeNYC --device 0
   ```

 - Check the output results (RMSE and MAE). Models are saved to "Exps" folder for further use.

## Test:
To obtain the best results reported in our paper, you can test our MUSE-Net via pre-trained models which can be downloaded from [BaiduYun](https://pan.baidu.com/s/1zeXvNfDu1BbDvgqcC7HupQ), password: tgoh.

 - Download the pre-trained models and put them in 'Exps' folder

 - Run with "python test.py" for NYC-Bike dataset, or "python test.py --dataset BikeNYC --device 0" for NYC-Bike dataset using GPU device 0

   ```
   python test.py
   ```

   ```
   python test.py --dataset BikeNYC --device 0
   ```

 - Check the output results (RMSE and MAE).
