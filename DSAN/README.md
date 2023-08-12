# DSAN: Preserving Dynamic Attention for Long-Term Spatial-Temporal Prediction 
#### *by: Haoxing Lin, Rufan Bai, Weijia Jia, Xinyu Yang, Yongjian You


## Requirements:
- Python 3.6.0
- Tensorflow-gpu >= 2.3.0
- CudaToolKit >= 10.1
- Cudnn >= 7.6.5
- Numpy
- h5py == 2.10.0

## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1yU8418Up3hT-3yTzVJ9byg?pwd=vjgm), password: vjgm. 

We used three public datasets in this study:
- NYC-Bike
- NYC-Taxi

## Usage 
You can select one of several training modes:
 - Download the NYC-Bike and NYC-Taxi datasets and put them in "data" folder, respectively. Notably, each dataset contains well-splited train, val, and test sets.

 - Run with "python main_1gpu.py" for NYC-Bike dataset, or "python main.py --dataset bike --gpu_ids 0" for NYC-Bike dataset using GPU device 0

   ```
   python main_1gpu.py
   ```

   ```
   python main_1gpu.py --dataset bike --gpu_ids 0
   ```
 
 - Run with "python main.py --dataset taxi --gpu_ids 0" for NYC-Taxi dataset using GPU device 0

   ```
   python main_1gpu.py --dataset bike --gpu_ids 0
   ```

 - Check the output results (RMSE and MAE) from "results" folder. Models are saved to "checkpoints" folder for further use. Output prediction adn labels are saved to "outputs" folder for further use.