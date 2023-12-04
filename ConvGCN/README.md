# Multi-graph convolutional network for short-term passenger flow forecasting in urban rail transit
#### *by: Jinlei Zhang, Feng Chen*, Yinan Guo, and Xiaohong Li.


## Requirements:
- Python
- Keras == 2.2.4
- tensorflow-gpu == 1.10.0
- numpy == 1.14.5
- scipy == 1.3.3
- scikit-learn == 0.20.2
- protobuf == 3.6.0  

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1WFhTG5KqIzJ-UzB3SmNKOQ?pwd=hm21). 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi
- TaxiBJ

## Usage 
You can select one of several training modes:
 - Download the datasets and put them in "Data" folder

 - Run with "python main.py" for NYC-Bike dataset, or "python main.py --dataset TaxiNYC --device 1" for NYC-Taxi dataset using GPU device 1

   ```
   python main.py
   ```

   ```
   python main.py --dataset TaxiNYC --device 1
   ```

 - Check the output results (RMSE and MAE). Models are saved to "exps" folder for further use.
