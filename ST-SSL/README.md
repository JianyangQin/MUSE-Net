# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction
#### *by: Ji, Jiahao and Wang, Jingyuan and Huang, Chao and Wu, Junjie and Xu, Boren and Wu, Zhenhe and Zhang Junbo and Zheng, Yu


## Requirements:
- Python 3.8
- PyTorch >= 1.10.1
- Numpy==1.21.2
- h5py == 2.9.0
- PyYAML==6.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1WFhTG5KqIzJ-UzB3SmNKOQ?pwd=hm21). 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ


## Usage 
You can select one of several training modes:
 - Download the NYC-Bike, NYC-Taxi, and TaxiBJ datasets and put them in "data/BikeNYC", "data/TaxiNYC", and "data/TaxiBJ" folders, respectively

 - Generate train/val/test datasets by running with "python generate.py" for NYC-Bike dataset, or "python generate.py --dataset TaxiNYC" for NYC-Taxi dataset

   ```
   python generate.py
   ```

   ```
   python generate.py --dataset TaxiNYC
   ```

 - Run with "python main.py" for NYC-Bike dataset, or "CUDA_VISIBLE_DEVICES=1 python train.py --config_filename configs/TaxiNYC.yaml" for NYC-Taxi dataset using GPU device 1

   ```
   python main.py
   ```

   ```
   CUDA_VISIBLE_DEVICES=1 python main.py --config_filename configs/TaxiNYC.yaml
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.

## Usage
You can select one of several training modes:
 - Download the NYC-Bike, NYC-Taxi and TaxiBJ datasets and put them in "data/BikeNYC", "data/TaxiNYC" and "data/TaxiBJ" folder, respectively

 - Run with "python train.py" for NYC-Bike dataset, or "python main.py --dataset BikeNYC --ctx 0" for NYC-Bike dataset using GPU device 0

   ```
   python train.py
   ```

   ```
   python train.py --dataset BikeNYC --ctx 0
   ```

 - Check the output results (RMSE and MAE). Models are saved to "experiments" folder for further use.

