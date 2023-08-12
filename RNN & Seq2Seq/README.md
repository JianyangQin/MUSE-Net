# RNN & Seq2Seq

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
You can test the RNN or Seq2Seq model:
 - Download the NYC-Bike, NYC-Taxi and TaxiBJ datasets and put them in "data" folder

 - Run with "python test.py --model_name RNN --dataset BikeNYC --ctx 0 --save_path exps" to test the RNN on the NYC-Bike dataset using GPU device 0

   ```
   python test.py --dataset BikeNYC --ctx 0 --save_path exps
   ```
   
 - Run with "python test.py --model_name Seq2Seq --dataset BikeNYC --ctx 0 --save_path exps" to test the Seq2Seq on the NYC-Bike dataset using GPU device 0

   ```
   python test.py --dataset BikeNYC --ctx 0 --save_path exps
   ```

 - Check the output results (RMSE and MAE).