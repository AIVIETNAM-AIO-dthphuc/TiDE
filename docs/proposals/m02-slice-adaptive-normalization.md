# feat/m02 — Slice-level Adaptive Normalization (SAN)

> Trạng thái: **đề xuất, chưa cài đặt**. Nhánh dự kiến: `feat/m02` (tách từ `main`). Ngày đề xuất: 2026-11-22.

## Tên phương pháp
Slice-level Adaptive Normalization — chuẩn hoá theo lát thời gian + dự đoán thống kê tương lai.
Tham khảo: Liu et al., *"Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice
Perspective"*, NeurIPS 2023.

## Thành phần TiDE liên quan
Khối reversible transform kiểu RevIN trong `TideModel.call` (`models.py`), bật bằng cờ `--transform`:
- Đầu vào: chuẩn hoá `past_ts` bằng `batch_mean`, `batch_std` của **toàn bộ lookback**, rồi affine
  (`affine_weight`, `affine_bias` theo từng series).
- Đầu ra: khử chuẩn hoá bằng **chính thống kê quá khứ** đó.

Theo `scripts/`: `--transform=true` ở ETTh1, ETTh2, ETTm2, Traffic; `false` ở Electricity, ETTm1, Weather.

## Vấn đề cải thiện
RevIN ngầm giả định:
1. Thống kê **không đổi trong lookback** — sai với cửa sổ 720 bước (≈30 ngày dữ liệu giờ).
2. Thống kê **tương lai = quá khứ** — khi có xu hướng / đổi chế độ (ETTh2, ETTm2), mức nền khôi phục ở đầu ra
   bị lệch; sai số lệch tăng theo H và phần MLP không có cơ chế sửa.

## Ý tưởng cải tiến
1. **Chuẩn hoá theo lát**: chia lookback thành các lát không chồng lấn dài T (vd T = 24 cho dữ liệu giờ,
   96 cho dữ liệu 15 phút); mỗi lát chuẩn hoá bằng μ, σ riêng.
2. **Mạng dự đoán thống kê**: MLP nhỏ (2 tầng) nhận chuỗi (μ, σ) các lát quá khứ (có thể kèm toàn cục) →
   dự đoán (μ̂, σ̂) cho từng lát trong horizon H. Có thể thêm residual từ thống kê lát cuối để ổn định.
3. **Khử chuẩn hoá theo lát**: mỗi lát đầu ra TiDE được khôi phục bằng (μ̂, σ̂) dự đoán cho lát đó.
4. **Huấn luyện**: loss phụ = MSE giữa (μ̂, σ̂) và (μ, σ) thật của từng lát tương lai.
   SAN gốc train 2 giai đoạn (mạng thống kê trước, backbone sau); đơn giản hơn là train chung với trọng số λ.

Cờ mới: `--san` (bool, mặc định false ⇒ giữ nguyên nhánh RevIN gốc), `--san_slice_len`, `--san_lambda`.

## Cơ sở khoa học
- Dịch chuyển phân phối (non-stationarity) là nguồn sai số chính của dự báo dài hạn.
- RevIN (Kim et al., ICLR 2022) chỉ xử lý dịch chuyển **giữa các cửa sổ**.
- Dish-TS (Fan et al., AAAI 2023): thống kê **horizon** khác lookback, cần ước lượng riêng.
- SAN mở rộng tới dịch chuyển **bên trong cửa sổ** ở mức lát; báo cáo cải thiện so với RevIN khi gắn vào
  nhiều backbone (DLinear, FEDformer, Autoformer…) trên ETT/Weather/Electricity/Traffic.

## Cơ chế kỳ vọng
- Tách nhiệm vụ: mạng thống kê lo **mức nền & biên độ thay đổi theo thời gian**; backbone TiDE chỉ học
  **hình dạng đã chuẩn hoá cục bộ** (dừng hơn) → dễ học, ít overfit trên dữ liệu nhỏ như ETT.
- Kỳ vọng cải thiện rõ ở dataset có xu hướng / trôi mức nền (**ETTh2, ETTm2**) và H dài — nơi RevIN yếu nhất.
- Có thể thử bật ở các dataset đang `--transform=false` (Weather, ETTm1).
- Rủi ro: thêm 2 siêu tham số; lát quá ngắn ⇒ thống kê nhiễu → chọn T bằng một chu kỳ tự nhiên.

## Ghi chú cài đặt
- Sửa trong `TideModel.__init__` (thêm MLP thống kê) và `TideModel.call` (thay khối `if self.transform:`
  ở cả đầu vào lẫn đầu ra) trong `models.py`.
- Yêu cầu `hist_len % T == 0` và `pred_len % T == 0` (hoặc xử lý lát cuối lẻ) — kiểm tra cấu hình
  scripts: hist_len ∈ {336, 720}, pred_len ∈ {96, 192, 336, 720}; T = 24 chia hết tất cả, T = 96 không chia hết 336.
- σ: kẹp tối thiểu bằng EPS (giống xử lý `batch_std == 0` hiện tại).
- Loss phụ cần thống kê lát tương lai từ `ytrue` → sửa `train_step` để cộng `λ · stat_loss`
  (`call` có thể trả thêm (μ̂, σ̂) khi training, hoặc lưu vào thuộc tính).
  `train_step` và `call` là `@tf.function` — chú ý shape tĩnh.
- Quyết định với affine RevIN hiện có: giữ affine theo series sau chuẩn hoá lát (khuyến nghị) để so sánh rõ.
- Đánh giá/early stopping giữ nguyên (`rmse` validation, MSE/MAE test).

## Đánh giá
- Baseline A: `--transform=false`; Baseline B: `--transform=true` (RevIN); Đề xuất: `--san=true`.
- Cùng siêu tham số `scripts/`, ≥3 `--random_seed`, H ∈ {96, 192, 336, 720}; báo MSE/MAE trung bình ± độ lệch chuẩn.
- Ưu tiên: ETTh2, ETTm2 (non-stationary rõ), sau đó ETTh1, Weather.
- Ablation: T ∈ {12, 24, 48}; λ ∈ {0.1, 0.5, 1.0}; train chung vs 2 giai đoạn.

## Không trùng với các hướng nhóm đã thử
(1)–(8) đều là thay đổi bên trong kiến trúc (gate, residual, decoder, Time2Vec, VSN). m02 thay khối chuẩn hoá
vào/ra (reversible normalization) — chưa ai đụng tới.
