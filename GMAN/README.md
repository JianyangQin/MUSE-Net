# GMAN: A Graph Multi-Attention Network for Traffic Prediction
#### *by: Chuanpan Zheng, Xiaoliang Fan*, Cheng Wang, and Jianzhong Qi


## Requirements:
- Python
- PyTorch
- CudaToolKit
- numpy
- h5py == 2.9.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1zeXvNfDu1BbDvgqcC7HupQ), password: tgoh. 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Usage 
You can select one of several training modes:
 - Download the NYC-Bike, NYC-Taxi, and TaxiBJ datasets and put them in "Data/BikeNYC", "Data/TaxiNYC", and "Data/TaxiBJ" folders, respectively

 - Run with "python main.py" for NYC-Bike dataset, or "python main.py --dataset TaxiNYC --device 1" for NYC-Taxi dataset using GPU device 1

   ```
   python main.py
   ```

   ```
   python main.py --dataset TaxiNYC --device 1
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.
