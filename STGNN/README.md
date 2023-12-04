# Traffic Flow Prediction via Spatial Temporal Graph Neural Network 
#### *by: Liangzhe Han, Bowen Du, Leilei Sun*, Yanjie Fu, Yisheng Lv, and Hui Xiong


## Requirements:
- Python 3.7.13
- PyTorch == 1.7.0
- CudaToolKit == 9.2
- numpy
- h5py == 2.9.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1WFhTG5KqIzJ-UzB3SmNKOQ?pwd=hm21). 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Usage 
You can select one of several training modes:
 - Download the datasets and put them in "Data" folder

 - Run with "python train.py" for NYC-Bike dataset, or "python train.py --dataset TaxiNYC --device 1" for NYC-Taxi dataset using GPU device 1

   ```
   python train.py
   ```

   ```
   python train.py --dataset TaxiNYC --device 1
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.

