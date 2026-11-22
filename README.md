# FEAT/M10: Cổng lọc biến nhẹ (Lightweight VSN Gate) cho TiDE (`feat/m10`)

## Sơ lược ý tưởng cải tiến
- **Thành phần tác động:** Feature Projection (`self.time_encoder` trong `models.py`) và cấu hình siêu tham số trong `train.py`.
- **Vấn đề trong TiDE gốc:**
  1. Khối `self.time_encoder` dùng trọng số tĩnh nén đồng đều mọi biến covariates về $\tilde{r}=4$ cố định (bài báo gốc chưa đánh giá ảnh hưởng của $\tilde{r}$), không có cơ chế lọc biến dư thừa hoặc nhiễu ngoại lai.
  2. Trên các bộ dữ liệu thực nghiệm của nhóm, nhiều cột covariates mang tính hằng số hoặc lệch địa lý (như `moh = -0.5` trên các tập tần suất `1H`, cột đệm `ncol = 0`, và 18 cột ngày lễ Mỹ bật trên 4 bộ ETT đo tại Trung Quốc).
- **Giải pháp (M10):**
  1. Bổ sung lớp `LightweightVSNGate` dạng bottleneck ($r \to r_g \to r$, mặc định $r_g = 8$) ngay trước `self.time_encoder`:
     $$\mathbf{g}_t = 2\sigma\big(\text{ReLU}(\mathbf{u}_t\mathbf{W}_{g1} + \mathbf{b}_{g1})\mathbf{W}_{g2} + \mathbf{b}_{g2}\big), \qquad \hat{\mathbf{u}}_t = \mathbf{g}_t \odot \mathbf{u}_t$$
     khởi tạo $\mathbf{W}_{g2} = \mathbf{0}, \mathbf{b}_{g2} = \mathbf{0} \implies \mathbf{g}_t = \mathbf{1}$ tại epoch 0 nhằm bảo toàn nguyên vẹn hành vi TiDE gốc lúc bắt đầu huấn luyện.
  2. Mở cờ `--temporal_width` (mặc định `4`) cho phép thực nghiệm kiểm chứng các giá trị $\tilde{r} \in \{2, 4, 8\}$ trực tiếp từ dòng lệnh/notebook.
- **Chi phí tham số & tốc độ:** Rất thấp (`< 0.01%` toàn mô hình), tính 1 lần trước `tf.repeat` nên giữ nguyên độ phức tạp $O(L)$.
- **Các file chỉnh sửa:** `models.py`, `train.py`, `README.md`.
- **Tài liệu tham khảo:** Lim et al. (2021) *Temporal Fusion Transformers (VSN/GRN)* (`arXiv:1912.09363`); Hu et al. (2018) *SE-Net* (`arXiv:1709.01507`); Das et al. (2023) *TiDE*.


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

