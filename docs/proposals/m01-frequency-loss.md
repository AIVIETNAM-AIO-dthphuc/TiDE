# feat/m01 — Frequency-Enhanced Loss (FreDF-style)

> Trạng thái: **đề xuất, chưa cài đặt**. Nhánh dự kiến: `feat/m01` (tách từ `main`). Ngày đề xuất: 2026-11-22.

## Tên phương pháp
Frequency-Enhanced Loss — hàm mất mát kết hợp miền thời gian + miền tần số.
Tham khảo: Wang et al., *"FreDF: Learning to Forecast in the Frequency Domain"*, ICLR 2025.

## Thành phần TiDE liên quan
Hàm mất mát huấn luyện: `train_loss = keras.losses.MeanSquaredError()` dùng trong
`TideModel.train_step` (`models.py`). **Không đổi kiến trúc.**

## Vấn đề cải thiện
- TiDE dự báo trực tiếp cả H bước (tới 720), nhưng MSE coi các bước như **độc lập**, trong khi chuỗi
  nhãn có tự tương quan rất mạnh giữa các bước liền kề.
- FreDF chỉ ra: tối ưu MSE miền thời gian khi nhãn tự tương quan là ước lượng **bị lệch** → dự báo bị
  **làm trơn quá mức** (mất biên độ chu kỳ ngày/tuần), nặng hơn khi H lớn.
- Residual tuyến tính `self.linear` + decoder MLP của TiDE càng dễ cho đầu ra trơn.

## Ý tưởng cải tiến
Thêm số hạng sai lệch phổ giữa dự báo và nhãn (độ dài H):

```
L = (1 − α) · MSE(ŷ, y) + α · mean | rFFT(ŷ) − rFFT(y) |
```

- `tf.signal.rfft` trên trục horizon; L1 trên hiệu phức (độ lớn `tf.abs` của hiệu) vì biên độ các tần số
  chênh nhau nhiều bậc, L1 ít bị vài tần số lớn chi phối.
- Cờ mới `--freq_loss_alpha` (mặc định `0.0` ⇒ trùng baseline). Quét α ∈ {0.25, 0.5, 0.75, 1.0} trên validation.

## Cơ sở khoa học
- Biến đổi Fourier xấp xỉ **khử tương quan** các bước của chuỗi dừng → các hệ số tần số gần độc lập,
  nên loss trên chúng thoả giả định độc lập tốt hơn MSE theo bước (lập luận lý thuyết + thực nghiệm của FreDF
  trên ETT/Weather/Electricity/Traffic, nhiều backbone).
- Các mô hình miền tần số (FEDformer; FITS, ICLR 2024; FreTS, NeurIPS 2023) cho thấy cấu trúc chu kỳ của
  các bộ dữ liệu này biểu diễn gọn nhất trong miền tần số.

## Cơ chế kỳ vọng
- Số hạng tần số phạt trực tiếp **sai biên độ / sai pha** của các chu kỳ chính (24h, 168h) → mô hình giữ
  dao động thay vì làm trơn.
- Kỳ vọng giảm MSE/MAE, rõ nhất ở **H = 336, 720** và dataset có chu kỳ mạnh (Electricity, Traffic, ETTh).
- Rủi ro thấp: α = 0 là baseline. Không thêm tham số; chi phí FFT trên ≤720 điểm không đáng kể.

## Ghi chú cài đặt
- Sửa `TideModel.train_step` (`models.py`): tính loss mới thay `train_loss(ytrue, all_preds)`.
  `ytrue`/`all_preds` có shape `[batch, pred_len]` → `rfft` theo `axis=-1` (cast float32 nếu cần).
- Truyền α từ `train.py` (định nghĩa cờ cạnh các cờ khác, `flags.DEFINE_float`) vào model / train_step.
  Lưu ý `train_step` là `@tf.function` — truyền α như hằng Python lúc khởi tạo model là đơn giản nhất.
- **Không đổi** loss đánh giá: early stopping vẫn theo `rmse` validation; test vẫn báo MSE/MAE như cũ.
- Nếu muốn ghi lại α vào kết quả, thêm vào phần log/metadata hiện có trong `train.py`.
- Dễ ghép với nhánh khác (chỉ chạm loss).

## Đánh giá
- Giữ nguyên siêu tham số trong `scripts/`, thêm `--random_seed` (≥3 seed) và `--freq_loss_alpha`.
- So với baseline (α = 0) cùng seed; báo MSE/MAE trung bình ± độ lệch chuẩn cho H ∈ {96, 192, 336, 720}.
- Ưu tiên chạy thử: ETTh1, ETTh2 (nhỏ, nhanh), sau đó Weather/Electricity.

## Không trùng với các hướng nhóm đã thử
(1) gate Temporal Decoder/Global Residual, (2) Adaptive Global Residual, (3) Gated Residual MLPResidual,
(4) LRGR, (5) Low-rank decoder projection, (6) Temporal-smoothed decoder, (7) Time2Vec, (8) VSN gate —
tất cả là thay đổi kiến trúc; m01 chỉ thay hàm mất mát.
