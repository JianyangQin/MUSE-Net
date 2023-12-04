# ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting 
#### *by: Jinliang Deng* and Xiusi Chen and Renhe Jiang and Xuan Song and Ivor W. Tsang


## Requirements:
- Python 3.7
- PyTorch >= 1.4.0
- CudaToolKit >= 9.2
- Numpy
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

 - Run with "python main.py" for NYC-Bike dataset, or "python main.py --dataset TaxiNYC" for NYC-Taxi dataset

   ```
   python main.py
   ```

   ```
   python main.py --dataset TaxiNYC
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.