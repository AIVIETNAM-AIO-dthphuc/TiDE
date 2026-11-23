# M08a — Output-Level Temporal Smoother + FreDF for TiDE

## Sơ lược ý tưởng cải tiến
- **Vấn đề cần cải thiện:** Tầng chiếu cuối `self.linear` và đầu ra của TiDE dự báo từng bước $t=1 \dots H$ theo cơ chế One-Shot độc lập giữa các vị trí, dễ tạo dao động răng cưa giữa các bước liền kề; đồng thời hàm loss `MeanSquaredError` chuẩn chưa xử lý tự tương quan chuỗi nhãn (label autocorrelation).
- **Ý tưởng cải tiến:**
  1. Gắn lớp tích chập 1D theo chiều sâu siêu nhẹ **`DepthwiseConv1D`** (`kernel_size=5`, `padding='same'`, `depth_multiplier=1`, `use_bias=False`, `depthwise_initializer='zeros'`) theo cơ chế **residual identity** ngay lên đầu ra dự báo `out` theo trục horizon (chỉ **5 tham số** ở `kernel_size=5`, dùng chung mọi dataset).
  2. Kết hợp hàm loss miền tần số **FreDF** (`tf.signal.rfft` + chuẩn hóa Parseval $1/\sqrt{H}$ + L1 phức mượt) với trọng số `--freq_alpha=0.5`.
- **Cơ sở khoa học & Kỳ vọng:** Kế thừa ModernTCN (**ICLR 2024**) về tích chập tách kênh 1D bắt liên tục cục bộ theo thời gian và Wang et al. (**ICLR 2025** — *FreDF*) về khử tự tương quan sai số trong miền tần số. Khởi tạo `zeros` đảm bảo giữ nguyên điểm hội tụ của `RevIN + Linear` ở epoch 0 và tránh hoàn toàn overfitting trên các tập nhỏ như ETT.
- **Kết quả kiểm chứng nhanh trên Sandbox (`ETTh2 H=96, seed=1`):** Đạt **MAE = `0.333830` (`-0.74%`, tốt nhất trong các biến thể thử nghiệm)** và **MSE = `0.271697` (`+0.48%`)** so với baseline TiDE (`0.270388 / 0.336325`).
- **Cờ chạy dòng lệnh mới:** `--smooth_kernel_size=5` (`<=0` để tắt), `--freq_alpha=0.5` (trong `[0.0, 1.0]`; `0.0` để tắt).

---

# Long-term Forecasting with TiDE: Time-series Dense Encoder

Authors: Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen and Rose Yu

Paper link: https://arxiv.org/abs/2304.08424

### Abstract
> Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we  show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.


## Code Organisation
The model is implemented in `models.py`. The main training loop is in `train.py` where in the input flags are defined. The dataloader is implemented in `data_loader.py`. The file `time_features.py` contains some utilities to convert the datetime into numerical features.

## Running Default Experiments
All the experiments in the paper can be reproduced by running the scripts in the `scripts/` folder. The name of each file corresponds to each dataset. Each such script will run 4 experiments, one each for each prediction length in {96, 192, 336, 720}. Before we can run these experiments we need to prepare the dataset.

### Step 1: Download data for default experiments
Do the following from the base folder to prepare the datasets:
```
cd datasets
bash download_data.sh
```
The above script will download the dataset and also attempt to create the `results` folder if it does not exist.

### Step 2: Train and evaluate
For any of the datasets in the paper we need to run the corresponding script. Let us say we need to run the `electricity` experiments. We need to just run the following script.
```
bash scripts/electricity.sh
```
Note that if you have multiple GPU's you can select the GPU to run on by adding the `--gpu=<device_id>` flag to the run command.

## Customising Scripts for New Datasets
The script supports any dataset with global dynamic covariates that are available in the future or without any covariates. Currently this research implementation does not support other types of covariates like static attributes. The steps for running on your own dataset are:

### Step 1: Prepare dataset.
The dataset should be a csv file with a `date` column that contains the datetime of the time-series. The different time-series should be columns in the dataset. Numerical global dynamic covariates can also be columns; the same applies to categorical covariates. 

### Step 2: Add dataset to dictionary in train.py
Then add the dataset into the `DATA_DICT` in `train.py`. It should include the `data_path`, `freq` of time-series and `boundaries` of train set end, validation set end and test set end. If you do not want to use the default settings, you can specify `ts_cols`, `num_cov_cols` and `cat_cov_cols` in the dictionary entry.

### Step 3: Launch training job.
An example launch command would be:
```
python3 -m train \
--transform=false \
--layer_norm=true \
--normalize=false \
--dropout_rate=0.5 \
--batch_size=512 \
--hidden_size=1024 \
--num_layers=2 \
--hist_len=720 \
--pred_len=720 \
--dataset=<new-dataset-name> \
--decoder_output_dim=8 \
--final_decoder_hidden=64 \
--num_split=1 \
--learning_rate=0.0001
```

In the above command, `<new-dataset-name>` corresponds to the dictionary entry corresponding to your new dataset. The package does not include tools for hyper-parameter tuning which is left to the user.

