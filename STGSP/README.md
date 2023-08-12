# ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction 
#### *by: Liang Zhao and Min Gao* and Zongwei Wang


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
 - Download the NYC-Bike, NYC-Taxi and TaxiBJ datasets and put them in "data/BikeNYC", "data/TaxiNYC" and "data/TaxiBJ" folder, respectively

 - Run with "python train.py" for NYC-Bike dataset, or "python main.py --dataset BikeNYC --ctx 0" for NYC-Bike dataset using GPU device 0

   ```
   python train.py
   ```

   ```
   python train.py --dataset BikeNYC --ctx 0
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.