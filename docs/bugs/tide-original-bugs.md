# Hai lỗi LayerNorm và dropout trong mã TiDE gốc

| Lỗi | Nguyên nhân | Biểu hiện trong mã Google gốc (`b338a83`) | Biểu hiện trên `main` (`00bd321`) | Cờ sửa lỗi |
| --- | --- | --- | --- | --- |
| LayerNorm ở `final_decoder` | Chuẩn hóa trục cuối có đúng một phần tử | Với `--layer_norm=true`, đầu ra của nhánh MLP luôn là `beta` | Cùng lỗi, chỉ `self.linear`, `beta` cuối và các tham số RevIN (nếu bật) có thể học | `--fix_final_decoder_ln` |
| Chế độ dropout bị giữ trong trace | `TideModel.call` dùng `@tf.function` và thiếu tham số `training` | Với `--dropout_rate` dương, trace đầu từ `train_step` bật dropout cả khi huấn luyện lẫn khi đánh giá | Lượt tạo biến dùng `training=False`, nên dropout tắt cả khi huấn luyện lẫn khi đánh giá | `--fix_dropout` |

`models.py` trên `main` giống `models.py` ở `b338a83`. Khác biệt về dropout đến từ thứ tự gọi trong `train.py`.

## 1. LayerNorm triệt tiêu gradient của nhánh MLP

Trong `main:models.py`, `TideModel.__init__` (dòng 150–155) tạo temporal decoder như sau.

```python
self.final_decoder = MLPResidual(
    hidden_dim=model_config.get('final_decoder_hidden'),
    output_dim=1,
    layer_norm=layer_norm,
    dropout_rate=dropout_rate,
)
```

`MLPResidual.__init__` tạo `LayerNormalization()` với trục mặc định `axis=-1`. `MLPResidual.call` (dòng 53–61) áp dụng lớp này sau khi cộng hai nhánh.

```python
if self.layer_norm:
  return self.lnorm(out + res)
```

Tensor đi ra từ `final_decoder` có dạng `B × H × 1`. Trên trục cuối, trung bình bằng chính giá trị cần chuẩn hóa và phương sai bằng 0. Thành phần chuẩn hóa luôn bằng 0, nên kết quả luôn bằng tham số `beta` của lớp này, khởi tạo bằng 0 và dùng chung cho mọi mẫu và bước dự báo tại một thời điểm huấn luyện.

Gradient từ loss về đầu vào của LayerNorm cuối bằng 0. Vì vậy `encoder`, `decoder`, `time_encoder`, các lớp Dense trong `final_decoder` và các embedding `ts_embs`, `cat_embs` đều có gradient bằng 0. `TideModel.call` cộng `self.linear(past_ts)` sau `final_decoder`, nên chỉ kernel và bias của `self.linear`, `beta` cuối và `affine_weight` / `affine_bias` của RevIN có thể học. Với `--transform=true`, mô hình thực chất là RevIN + Linear với một độ lệch vô hướng từ `beta`. Với `--transform=false`, mô hình chỉ còn phép dự báo tuyến tính kèm độ lệch. Xem `main:models.py`, dòng 112–125 và 180–218.

Các script trong `scripts/` đều đặt `--layer_norm=true`, riêng `traffic.sh` đặt `--layer_norm=false`. Khi đo trên CPU với tensor tổng hợp và `--transform=true`, **33/38 biến trainable có gradient bằng đúng 0**. Các test `test_original_bug_dead_backbone` và `test_fix_restores_backbone_gradients` kiểm tra gradient trước và sau khi sửa. Trong các lần chạy ETTh2 trước bản sửa, nhóm ghi nhận hai biến thể M07 và M08 (thay đổi các khối trong nhánh MLP, như dense encoder/decoder) có metric trùng baseline đến 15 chữ số thập phân. Cả hai thay đổi nằm trên đường đi bị LayerNorm cuối triệt tiêu. [Các script gốc](../../scripts/) · [Kiểm tra gradient](../../tests/test_bug_fixes.py#L120-L140)

## 2. Lần gọi đầu cố định chế độ dropout

Trong `main:models.py`, `TideModel.call` (dòng 171–172) chỉ nhận `inputs`.

```python
@tf.function
def call(self, inputs):
```

`MLPResidual.call` gọi `self.dropout(out)` (dòng 57) và lấy chế độ huấn luyện từ ngữ cảnh Keras. `TimeSeriesdata.tf_dataset` trong `data_loader.py` gọi `tf.data.Dataset.from_generator(gen_fn, output_types)` với cùng bộ kiểu dữ liệu cho train, val và test, không khai báo shape. Khi các batch có cùng chữ ký tensor, các lượt gọi dùng chung trace của `TideModel.call`. Chế độ huấn luyện được đọc lúc tạo trace và giữ trong graph vì `training` vắng mặt trong chữ ký. [TimeSeriesdata.tf_dataset](../../data_loader.py)

- **Mã Google gốc.** `training()` gọi `model.train_step` trước khi chạy val/test (`b338a83:train.py`, dòng 228–250). `train_step` gọi model với `training=True` (`b338a83:models.py`, dòng 221–229). Với `--dropout_rate` dương, trace đầu áp dụng dropout cả khi các lượt đánh giá sau đó truyền `training=False`, làm dự báo val/test dao động nếu nhánh MLP còn ảnh hưởng đến đầu ra.
- **`main` của repo.** `training()` gọi `_build_model_and_optimizer()` trước vòng lặp huấn luyện, kể cả khi chạy mới. Hàm này lấy một batch train và gọi `model(..., training=False)` để tạo biến trước khi khôi phục checkpoint (`main:train.py`, dòng 213–223 và 368–385). Trace tạo khi đó tắt dropout. Các batch huấn luyện dùng lại trace này, nên giá trị `--dropout_rate` không có tác dụng.

Kiểm tra trên CPU với `main:models.py`, `layer_norm=False`, `dropout_rate=0.5` và `lr=0` cho cùng một batch tổng hợp. Theo thứ tự gọi của `main`, cả ba loss từ `train_step` bằng nhau và các dự báo eval giống nhau. Theo thứ tự gọi của mã Google gốc, loss dao động giữa các bước và dự báo eval cũng dao động. [Hàm tạo batch và lặp phép đo](../../tests/test_bug_fixes.py#L45-L99)

## Tương tác giữa hai lỗi

Với `--layer_norm=true` và LayerNorm cuối còn lỗi, mọi nhiễu do dropout trong nhánh MLP đều bị xóa trước khi cộng `self.linear(past_ts)`. Trong phép thử tổng hợp theo thứ tự gọi của Google, bật lại `layer_norm=True` làm các loss và dự báo eval lặp lại giống nhau. Điều này giải thích vì sao khác biệt về dropout giữa hai mã vẫn cho phép baseline `main` của nhóm khớp kết quả bài báo trên ETT. Với `traffic.sh`, `--layer_norm=false` giữ nhánh MLP hoạt động, nên sai khác về dropout có thể ảnh hưởng đến kết quả. [Đường đi của dự báo trong TideModel.call](../../models.py#L219-L234) · [Cấu hình traffic.sh](../../scripts/traffic.sh)

## Bản sửa và cách chạy trên Kaggle

- `--fix_final_decoder_ln=true` là mặc định. `TideModel.__init__` dùng `layer_norm=layer_norm and not fix_final_decoder_ln` riêng cho `final_decoder`. `--layer_norm` vẫn điều khiển các khối khác. [TideModel.__init__](../../models.py#L143-L171)
- `--fix_dropout=true` là mặc định. `TideModel.call(self, inputs, training=None)` tạo trace riêng cho `training=True` và `training=False` với cùng chữ ký tensor. Dropout áp dụng khi huấn luyện nếu `--dropout_rate` dương và tắt khi đánh giá. Với `--fix_dropout=false`, hàm khởi tạo ép `dropout_rate = 0.0` để tái lập hành vi của `main`. [Cấu hình dropout](../../models.py#L123-L127) · [TideModel.call](../../models.py#L187-L195) · [Cờ CLI](../../train.py#L73-L84)
- `_run_config()` chỉ ghi khóa của bản sửa đang bật vào `run_config.json`. `_verify_or_write_config()` từ chối cấu hình khác nhau trước khi nạp trạng thái TensorFlow. Khi cả hai cờ là `false`, các khóa khớp định dạng cũ. Test `test_checkpoint_from_base_code_restores_and_matches` nạp checkpoint mô hình từ `00bd321` và so sánh dự báo ở cả hai chế độ, với cả hai giá trị `layer_norm`. [Kiểm tra cấu hình](../../train.py#L173-L213) · [Test checkpoint](../../tests/test_bug_fixes.py#L207-L268) · [Các test cấu hình](../../tests/test_bug_fixes.py#L271-L366)

Ví dụ chạy baseline ETTh2 đã sửa trên Kaggle với horizon 96, seed 1 và các tham số mô hình từ `scripts/etth2.sh`. Notebook của nhóm ghi đè `--min_num_epochs` từ `0` thành `20`. [Cấu hình ETTh2](../../scripts/etth2.sh)

Tạo trước một thư mục Drive riêng cho mỗi phương pháp hoặc cấu hình và truyền đường dẫn qua `--remote_base`. Nếu dùng lại thư mục có snapshot của một cấu hình khác, phép đối chiếu `run_config.json` sẽ từ chối khôi phục trạng thái huấn luyện. Khi chuyển cấu hình, dùng phiên Kaggle mới nếu phiên hiện tại còn checkpoint cục bộ cho cùng dataset, seed và horizon, vì `--resume=true` ưu tiên checkpoint cục bộ. [Chọn thư mục và checkpoint](../../train.py#L273-L319)

```bash
python -m train \
  --dataset=etth2 --random_seed=1 --pred_len=96 \
  --transform=true --layer_norm=true --holiday=true \
  --dropout_rate=0.2 --batch_size=512 --hidden_size=512 --num_layers=2 \
  --hist_len=720 --decoder_output_dim=32 --final_decoder_hidden=16 \
  --num_split=1 --learning_rate=0.00022400011085303776 \
  --min_num_epochs=20 --train_epochs=100 --patience=40 \
  --fix_final_decoder_ln=true --fix_dropout=true \
  --expt_dir=/kaggle/working/baseline_fixed \
  --drive_backup=true --resume=true --backup_every=1 --max_to_keep=2 \
  --remote_base="gdrive:[54] IS403 - BA/[1] PROJECT/[IS403]_Experiment/baseline_fixed"
```

Để tái lập cấu hình hoặc tiếp tục một lần chạy cũ từ `main`, dùng lại thư mục Drive ban đầu qua `--remote_base` và đặt `--drive_backup=true --resume=true --fix_final_decoder_ln=false --fix_dropout=false`. Giữ cùng dataset, `--random_seed`, `--pred_len` và các siêu tham số đã lưu, kể cả giá trị `--dropout_rate` ban đầu. Nếu cần tái lập dropout luôn bật của mã Google gốc, dùng `models.py` ở `b338a83` theo thứ tự gọi `train_step` trước lượt suy luận. [Tương thích cấu hình cũ](../../README.md#bug-fixes-in-this-branch) · [Luồng khôi phục checkpoint](../../train.py#L298-L319)

### Cách kiểm tra

Sau khi notebook clone nhánh, đối chiếu mã commit notebook in ra (`git rev-parse HEAD`) với mã commit dự kiến cho lần chạy.

Tìm dòng log sau khi khởi động ví dụ baseline đã sửa. Khi tái lập `main`, hai giá trị cờ trên dòng này đều là `False`. [Dòng log trong training()](../../train.py#L411-L416)

```text
Run: dataset=etth2 seed=1 H=96 fix_final_decoder_ln=True fix_dropout=True local=<đường dẫn checkpoint> remote=<đường dẫn snapshot trên Drive>
```

Để kiểm tra dropout trên CPU, đặt `TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1`, dùng cùng một batch tensor tổng hợp cho ba lần `train_step` với `SGD(0.0)` và ba lượt eval. Đặt `dropout_rate=0.5` và giữ nhánh MLP hoạt động bằng cách tắt LayerNorm hoặc bật bản sửa LayerNorm. Tạo model mới cho mỗi thứ tự gọi, một model chạy `training=False` trước và một model vào thẳng `train_step`. Với cả hai bản sửa, loss huấn luyện có thể dao động, còn các dự báo eval phải trùng nhau ở cả hai thứ tự. [Các test dropout](../../tests/test_bug_fixes.py#L153-L193)

## Trạng thái các nhánh

| Nhánh | Sửa LayerNorm cuối | Sửa dropout |
| --- | --- | --- |
| `fix-LN-n-dropout` (nhánh cơ sở của `task/bugnote`) | Có, mặc định `true` | Có, mặc định `true` |
| `feat/m01_fixLN` | Có, `--fix_final_decoder_ln` mặc định `true` (`a493fd3`) | Chưa có |
| `feat/m02` | Có, `--fix_final_decoder_ln` mặc định `true` (`d55e737`, `e60cad8`) | Chưa có, `73f5f53` giữ hành vi dropout cũ |
| `feat/m01` | Chưa có | Chưa có |
| `main` (nhánh được bảo vệ, tại `00bd321`) | Chưa có | Chưa có |

## Khuyến nghị khi so sánh kết quả

- Đọc baseline từ `main` với dropout hiệu dụng bằng 0. Với các lần chạy bật LayerNorm, kết quả phản ánh nhánh Linear, kèm RevIN nếu bật `--transform=true`. Traffic giữ nhánh MLP. Với LayerNorm lỗi, các cải tiến chỉ tác động lên phần MLP bị chặn sẽ cho cùng dự báo khi giữ nguyên các tham số còn tác dụng. [Đường đi của dự báo](../../models.py#L219-L234) · [Kiểm tra gradient](../../tests/test_bug_fixes.py#L120-L140)
- Chạy lại baseline với cả hai bản sửa trong thư mục Drive mới. Để so sánh công bằng, dùng cùng dataset, seed, horizon và điều kiện đánh giá cho baseline và biến thể.
- Theo ghi nhận của nhóm, các siêu tham số trong script cũ, như `--dropout_rate=0.3` hoặc `0.5`, `--learning_rate` và `--hidden_size`, được tinh chỉnh cho mã còn lỗi. Có thể cần thử dropout trong tập `{0.0, 0.3, 0.5}` rồi chọn theo tập val. Kết quả sau khi sửa có thể tốt hơn hoặc kém hơn số liệu trong bài báo. [Các cấu hình có sẵn](../../scripts/)
