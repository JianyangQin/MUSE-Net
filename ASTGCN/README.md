# ASTGCN: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
#### *by: Shengnan Guo and Youfang Lin and Ning Feng and Chao Song and Huaiyu Wan*


## Requirements:
- Python 3.7
- PyTorch >= 1.7.1
- CudaToolKit >= 10.2
- Numpy
- h5py == 2.9.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1zeXvNfDu1BbDvgqcC7HupQ), password: tgoh. 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Usage 
You can select one of several training modes:
 - Download the NYC-Bike, NYC-Taxi and TaxiBJ datasets and put them in "data" folder

 - Run with "python main.py" for NYC-Bike dataset, or "python main.py --config configurations/BikeNYC_astgcn.conf" for NYC-Bike dataset using pre-defined configuration

   ```
   python main.py
   ```

   ```
   python main.py --config configurations/BikeNYC_astgcn.conf
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.