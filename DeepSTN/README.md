# ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction 
#### *by: Liang Zhao and Min Gao* and Zongwei Wang


## Requirements:
- Python 3.6
- tensorflow-gpu ==1.3.0
- keras == 2.0.8
- Numpy
- Pandas
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

 - Run with "python main.py" for NYC-Bike dataset, or "python main.py --dataset BikeNYC --device 0" for NYC-Bike dataset using GPU device 0

   ```
   python main.py
   ```

   ```
   python main.py --dataset BikeNYC --device 0
   ```

 - Check the output results (RMSE and MAE). Models are saved to "Exps" folder for further use.