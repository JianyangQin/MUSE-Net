# Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting
#### *by: Liangzhe Han, Bowen Du, Leilei Sun*, Yanjie Fu, Yisheng Lv, and Hui Xiong


## Requirements:
- Python 3.7.4
- PyTorch == 1.2.0
- CudaToolKit == 9.2
- numpy == 1.17.2
- h5py == 2.9.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1WFhTG5KqIzJ-UzB3SmNKOQ?pwd=hm21). 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Usage 
You can select one of several training modes:
 - Download the NYC-Bike, NYC-Taxi, and TaxiBJ datasets and put them in "Data/BikeNYC", "Data/TaxiNYC", and "Data/TaxiBJ" folders, respectively

 - Generate train/val/test datasets by running with "python generate.py" for NYC-Bike dataset, or "python generate.py --dataset TaxiNYC" for NYC-Taxi dataset

   ```
   python generate.py
   ```

   ```
   python generate.py --dataset TaxiNYC
   ```

 - Run with "python train.py" for NYC-Bike dataset, or "python train.py --dataset TaxiNYC --device cuda:1" for NYC-Taxi dataset using GPU device 1

   ```
   python train.py
   ```

   ```
   python train.py --dataset TaxiNYC --device cuda:1
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.
