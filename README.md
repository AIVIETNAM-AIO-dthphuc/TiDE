# M09: Mã hóa thời gian Time2Vec cho TiDE (`feat/m09`)

## Sơ lược ý tưởng cải tiến
- **Thành phần tác động:** Input Data (`time_features.py`) và đầu vào Feature Projection (`self.time_encoder` trong `models.py`).
- **Vấn đề trong TiDE gốc:**
  1. Chuẩn hóa tuyến tính về `[-0.5, 0.5]` cho 7 đặc trưng chu kỳ thời gian tạo ra điểm nhảy gián đoạn ở biên chu kỳ (ví dụ `23:00 = +0.5` nhảy về `00:00 = -0.5`) và lỗi lệch 1-indexed ở đặc trưng ngày/tháng (`month / 11.0 - 0.5` đạt `+0.591`).
  2. Mạng MLP tiêu chuẩn chịu thiên kiến phổ (*spectral bias*), học chậm các biến thiên tuần hoàn tần số cao từ tọa độ thời gian tuyến tính.
- **Giải pháp (M09):**
  1. Chuẩn hóa lại 7 đặc trưng chu kỳ trong `time_features.py` theo đúng chu kỳ thực $P$ (bảo toàn khoảng cách đều trên đường tròn đơn vị và sửa lỗi 1-indexed).
  2. Thêm lớp `Time2VecLayer` trong `models.py` ngay trước `self.time_encoder`, ánh xạ 7 biến chu kỳ đầu tiên thành 14 chiều liên tục học được $(\omega_0\tau + \varphi_0,\; \sin(\omega_1\tau + \varphi_1))$ (khởi tạo $\omega_0=1, \varphi_0=0, \omega_1=2\pi, \varphi_1=0$), giữ nguyên các cột ngày lễ/đệm phía sau, giữ nguyên $\tilde{r}=4$ và toàn bộ khối Encoder/Decoder.
- **Chi phí tham số:** Tăng thêm `+504` tham số (`+28` tham số Time2Vec và `+476` tham số đầu vào `time_encoder`, `<0.005%` toàn mô hình), giữ nguyên độ phức tạp $O(L)$.
- **Các file chỉnh sửa:** `time_features.py`, `models.py`, `train.py` (thêm cờ `--use_time2vec=true`), `README.md`.
- **Tài liệu tham khảo:** Kazemi et al. (2019) *Time2Vec* (`arXiv:1907.05321`); Tancik et al. (2020) *Fourier Features* (`arXiv:2006.10739`); Das et al. (2023) *TiDE*.

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

