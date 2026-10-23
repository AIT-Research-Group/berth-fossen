# Mô phỏng gió và sóng phản xạ theo khung Fossen

Mã nguồn nghiên cứu bằng Python, sinh dữ liệu cảm biến gió/sóng và đáp ứng tàu 3-DOF, kèm mô hình LSTM và conditional GAN–LSTM để huấn luyện dự báo xác suất. Đây chưa phải hệ thống đánh giá an toàn cập tàu.

## Chạy

Yêu cầu Python >= 3.10. Tại thư mục dự án:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install .
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m berth_sim --config configs/demo.json --output outputs/demo
```

Nếu dùng uv, thay bước cài đặt bằng `uv pip install --python .venv/Scripts/python.exe .`.
`PYTHONPATH=src` giúp chạy trực tiếp mã nguồn đang sửa. Cài dạng thường tránh lỗi mã hóa đường dẫn tiếng Việt của editable install trên một số bản Python/Windows.
Mỗi lần chạy dùng một thư mục đầu ra mới; chương trình từ chối ghi đè.

## Mô hình và quy ước

- Gió tuyệt đối: trung bình biến đổi chậm cộng nhiễu loạn phổ Kaimal, tổng hợp trên lưới FFT của cả chuỗi với hệ số phức Gauss (kiểu TurbSim) nên không lặp trong chuỗi. Hai thành phần nhiễu được sinh độc lập; chưa mô tả phổ chéo không gian hoặc che chắn công trình. Phiên bản trước dùng 256 điều hòa cách đều, lặp lại sau ~285 s; kết quả `training_v1`/`force_training_v1` bị ảnh hưởng và được thay bằng `training_v2`/`force_training_v2`.
- Gió tương đối: quay gió tuyệt đối vào hệ thân tàu rồi trừ vận tốc tàu. Tải khí động tỷ lệ bình phương tốc độ tương đối. Các luật hệ số lượng giác là **minh họa**, không phải hệ số đã nhận dạng cho tàu cụ thể.
- Sóng: phổ JONSWAP một phía theo rad/s, dạng DNV-RP-C205 với hệ số chuẩn hóa A_γ = 1 − 0,287 ln γ (1 ≤ γ ≤ 7; Hm0 sai < 1% so với Hs). Không ép chuẩn hóa lại trong dải mô phỏng; tỉ lệ năng lượng dải được lưu (`band_energy_fraction`) và từ chối cấu hình nếu dải giữ < 90% năng lượng. Tổng điều hòa pha ngẫu nhiên với tần số rút ngẫu nhiên trong từng ô (`frequency_jitter`) để không lặp chu kỳ 2π/Δω; tùy chọn biên độ Rayleigh (`random_amplitude`, Tucker và cs. 1984) chỉ nên dùng khi độ dài chuỗi cỡ 2π/Δω. Bước tần số phải ≤ 1/`feature_window_s`. Giải quan hệ tán sắc hữu hạn độ sâu.
- Hướng sóng: mặc định sóng tới vuông góc tường. `mean_direction_rad` cho sóng xiên (phản xạ gương), `spreading_s` bật phân bố hướng cos-2s dạng nửa góc (Longuet-Higgins và cs. 1963; Mitsuyasu và cs. 1975), mỗi tần số một hướng (single-summation), cắt bỏ hướng không đi vào tường. Đầu đo nằm trên đường y=0 vuông góc tường nên chỉ k·cosθ xuất hiện.
- Phản xạ: cùng pha nguồn với sóng tới, hệ số biên độ và độ lệch pha phản xạ. `reflection` là số (Kr cho trước, phù hợp tường đứng) hoặc `{"model": "zanuttigh_van_der_meer", "slope_tan": ..., "roughness_factor": ...}`: công thức thực nghiệm Kr = tanh(a ξ^b) theo Hs và T_{m−1,0} (Zanuttigh & van der Meer 2008) cho kết cấu mái dốc, có phân tán thực nghiệm, không dùng cho tường đứng. Đây là phần bổ sung biên tuyến tính vào khung Fossen; không mô phỏng cộng hưởng khe tàu–cầu.
- Tàu: `M nu_dot + C(nu) nu + D(nu) nu = tau`, tích phân RK4. M là tổng quán tính gồm khối lượng gia tăng, ma trận chéo minh họa. C phản đối xứng; D có phần tuyến tính và bậc hai. Không xét dòng chảy.
- Bộ điều khiển giữ vị trí PD có giới hạn lực, chỉ phục vụ chạy thử; không mô phỏng chân vịt, tàu lai hoặc quá trình điều động cập tàu.
- Lực sóng dùng force RAO tại vị trí/hướng tham chiếu cố định, tách hướng tới và phản xạ. API nhận RAO phức theo tần số; cấu hình demo dùng giá trị thực hằng. Chưa xét tần số gặp sóng khi tàu di chuyển, lực trôi bậc hai hay bộ nhớ bức xạ. Đáp ứng tàu là minh họa, không phải kết quả seakeeping đã hiệu chỉnh.

Hệ tọa độ thuận tay phải, z hướng lên: tường x=0, nước x<0, trục x hướng vào tường, y dọc cầu. Góc dương ngược chiều kim đồng hồ nhìn từ trên. Gió chỉ hướng vận tốc **đi tới**. Các hệ số NED/BODY từ nguồn khác phải đổi quy ước trước khi dùng. Sóng tới truyền +x, phản xạ -x. Trong bản thảo, thành phần song song cầu tương ứng y và thành phần pháp tuyến tương ứng x nếu chọn cùng chiều trục.

## Đầu ra

| File | Nội dung |
|---|---|
| `observations.npz` | Thời gian tương đối [s], gió tuyệt đối [m/s], tín hiệu các đầu đo sóng [m] |
| `truth.npz` | Sóng tới/phản xạ sạch, phổ gốc, hệ số phản xạ và bao biên |
| `features.csv` | Đặc trưng nhân quả, chất lượng phép tách và đặc trưng gió |
| `feature_truth.csv` | Hm0 tham chiếu từ phương sai sóng phản xạ sạch trong cửa sổ |
| `vessel.npz` | `[x,y,psi,u,v,r]`, lực gió/sóng BODY [N,N,Nm] |
| `metadata.json` | Cấu hình đầy đủ, seed, phiên bản, giới hạn mô hình, số mẫu |

`truth` là nghiệm mô phỏng để đánh giá; **không đưa vào đầu vào AI**. Trong mô phỏng ngắn, Hm0 cửa sổ từ phương sai mẫu không nhất thiết bằng đúng Hm0 phổ thiết kế.

Mặc định dt=0.25 s, 30 phút dữ liệu. Sóng dùng cửa sổ Hann lùi 10 phút, bước 1 phút; gió trung bình trong phút cuối; gió giật là cực đại trung bình trượt 3 giây của tốc độ trong phút đó. Nhãn thời gian là mẫu cuối thực sự đã sử dụng. Bản demo chỉ tạo 21 dòng đặc trưng, **không đủ huấn luyện AI**.

Tách sóng dùng least squares/SVD trên từng tần số và kiểm tra điều kiện ma trận. Chỉ xuất năng lượng toàn dải khi mọi ô tần số trong dải đều hợp lệ; nếu thiếu mẫu hoặc tách không ổn định thì Hm0 là NaN và `quality_ok=0`. Tỉ lệ tần số hợp lệ và phần dư vẫn được báo cáo. Phổ tách được lấy trung bình Welch: các đoạn Hann dài `welch_segment_s` chồng 50% (demo: 150 s, 7 đoạn trong cửa sổ 10 phút); tương đương áp toán tử least squares lên ma trận phổ chéo Welch. Trên 5 kịch bản demo 2 giờ, sai số tương đối Hm0 phản xạ có độ lệch chuẩn 2,5% so với 5,4% khi dùng một cửa sổ, độ chệch ≈ 0. `welch_segment_s: null` quay về một cửa sổ. Phương pháp giả định hướng tới trung bình đã biết (`mean_direction_rad`); trải hướng và rò phổ của cửa sổ Hann làm phần dư khác 0.

## Giao diện trực quan 3D

```powershell
.venv/Scripts/python.exe scripts/wave_viewer.py --port 8090
```

Mở http://127.0.0.1:8090 (thêm `?tab=spectrum|vessel|forecast` để mở thẳng một tab). Plotly.js 4.1.1 đặt sẵn trong `viewer/vendor`, chạy offline, server chỉ nghe trên 127.0.0.1. Mọi hình 3D có colormap parula/jet, xoay/zoom/datatip, xuất PNG ×3, và các góc nhìn 3D `view(−37.5°, 30°)`, trên xuống, mặt cắt.

| Tab | Nội dung | Nguồn |
|---|---|---|
| Mặt sóng η(x, t) | Mặt `surf` sóng tới/phản xạ/tổng, tường, đầu đo; lát cắt η(x, t*); Hm0(x) thể hiện nút/bụng sóng đứng | Dựng lại từ seed và cấu hình; trùng chuỗi đã lưu tới 1e−9 m |
| Phổ theo thời gian | Mặt S(f, t) đã tách bằng Welch cho từng cửa sổ; phổ một cửa sổ so với JONSWAP thiết kế và Kr² × thiết kế; Hm0 phản xạ thật và ước lượng, đánh dấu cửa sổ bị loại | Mô phỏng lại bằng chính `causal_features`, nên trùng đặc trưng huấn luyện |
| Biển & tàu | Mặt biển η(x, y, t) quanh tàu (sóng tới + phản xạ gương, có hướng/trải hướng), phát theo thời gian; tàu vẽ đúng tỉ lệ bằng thân Wigley với B/L, T/L của tàu dầu MSS; phổ JONSWAP đánh dấu tần số có λ = L, λ = B, fp và phần sóng quá ngắn không vẽ được trên lưới; η dọc trục tàu và ngang tàu tới tường | Dựng lại từ seed và cấu hình; nước hở, chưa tính nhiễu xạ/bức xạ quanh thân tàu; tàu đứng yên |
| Tàu & tải | Phát lại tàu 3 bậc tự do cạnh tường (thân tàu sơ đồ, phóng đại chuyển động), quỹ đạo, mũi tên lực gió/sóng/tổng; chuỗi tải X/Y/N và lệch vị trí có con trỏ thời gian | `outputs/*/vessel.npz` |
| Dự báo AI | Mặt dự báo trung bình, sai số hoặc độ rộng 90% theo chân trời × thời điểm phát hành, lưới quan trắc kiểu `mesh`; biểu đồ quạt với lịch sử 60 phút; RMSE và coverage theo chân trời | `outputs/*/evaluation/*_test.npz` và `dataset/test.npz`; không chạy suy luận mới |

## Kịch bản và giới hạn dữ liệu

`storm` tạo một xung Gaussian chậm cho tốc độ gió và biên độ sóng, có độ trễ riêng. Đây là kịch bản quy định trước, không phải luật vật lý gió phát sinh sóng. Bao biên sóng nhân đồng thời tại các vị trí là xấp xỉ gần dừng; không mô tả truyền bao sóng với vận tốc nhóm. Trễ nhóm trên quãng đường đầu đo–tường–đầu đo (~50 m) chỉ khoảng 10–15 s với Tp 4–8 s ở độ sâu 15 m, nhỏ hơn nhiều so với độ rộng xung bão (≥ 300 s).

Để xây dữ liệu nghiên cứu: tạo nhiều cấu hình độc lập về Hs, Tp, hướng gió, phản xạ, độ sâu, thời tiết và lỗi cảm biến; chia train/validation/calibration/test **theo kịch bản trước khi cắt cửa sổ**. Giữ riêng kịch bản ngoài miền huấn luyện. Không chỉ đổi seed rồi coi là kiểm chứng khái quát hóa vật lý.

Trước khi dùng kết quả tàu: thay bộ tham số tàu, hệ số gió và RAO bằng dữ liệu nhận dạng/tính thủy động lực; kiểm tra hội tụ thời gian và phạm vi nhỏ của chuyển động quanh điểm tham chiếu. Chưa có fender, tiếp xúc hoặc nhãn an toàn. Không diễn giải vị trí tàu như bằng chứng bảo đảm cập tàu an toàn.

## Huấn luyện mô hình dự báo

Bộ tải có thể dùng bảng RAO lực kích động tàu dầu công khai từ MSS/Fossen tại
`data/reference/mss_tanker/tanker.mat`. `hydrodynamics.TankerHydrodynamics` nội suy RAO phức theo tần số và
hướng, đổi từ NED/exp(+iωt) của MSS sang hệ BODY z-up/exp(−iωt) của mô phỏng (liên hợp phức, đổi dấu Y và N,
hướng là chiều sóng **đi tới** so với mũi tàu) và ghi hash nguồn vào manifest. Kiểm thử khóa quy ước bằng
đối xứng trái–phải, lực ngang bằng 0 khi sóng dọc tàu và pha −90° của lực Froude–Krylov sóng dài. Dữ liệu được
ghim tại commit MSS `a175d836808ff723924198fc4fad352054aa18e2`; đây là tàu dầu
benchmark dài 246 m, chưa phải hệ số đã hiệu chỉnh cho một cảng cụ thể.

### Dữ liệu lực và mômen

Khi `vessel.enabled=true`, lệnh mô phỏng còn xuất `load_features.csv`: trung bình có dấu, RMS và cực đại trị tuyệt đối của X, Y, N trong mỗi cửa sổ lùi dài `feature_step_s` (mặc định 60 giây). Tách riêng tải gió, tải sóng và tổng tải môi trường. Đơn vị lực N, mômen Nm; tất cả trong hệ BODY. Tổng tải được cộng ở cấp tín hiệu trước khi tính RMS, không cộng các RMS riêng lẻ. `vessel.npz` lưu thêm `environment_force_body` ở tần số mẫu gốc.

Đã chạy ví dụ tại `outputs/loads_demo`, kèm `load_rms.png`. Các nhãn tải là đầu ra mô phỏng chưa hiệu chỉnh, không phải số đo. RAO demo trong `configs/demo.json` cho X và N của sóng bằng 0; mômen gió vẫn có thể khác 0. Không tự thêm hệ số RAO không có căn cứ để làm các thành phần này khác 0.

Checkpoint `training_v1` chỉ dự báo môi trường. Phiên bản tải dùng `force_training_v1` riêng, không gán kết quả huấn luyện cũ thành dự báo lực.

### Huấn luyện dự báo tải

```powershell
$env:PYTHONPATH = 'src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tmp/matplotlib'
.venv/Scripts/python.exe -m berth_sim.training.force_run --config configs/force_training.json --output outputs/force_training_v1
```

Mô hình tải nhận **lịch sử sáu đặc trưng tải**: RMS X/Y/N và cực đại trị tuyệt đối X/Y/N. Đầu ra cũng là sáu đặc trưng này trong 30 phút tiếp theo. Đây là dự báo tải trực tiếp từ lịch sử tải mô phỏng, chưa phải truyền ensemble gió/Hm0 qua mô hình vật lý và chưa nhận riêng gió/sóng làm đầu vào. Để triển khai thực tế cần lịch sử tải đo/ước lượng có cùng định nghĩa.

56 kịch bản, mỗi kịch bản 4 giờ; tàu giữ cố định ở vị trí/hướng tham chiếu, vận tốc bằng 0 để phù hợp với force RAO cố định. Không tích phân chuyển động tàu trong bộ dữ liệu huấn luyện tải. Sinh tải sóng từ cùng trường sóng/pha của kịch bản, cộng tải gió tức thời trước khi lấy RMS/cực đại. Khi có `hydro_reference`, lực sóng dùng RAO tàu dầu MSS cho **cả sóng tới (hướng θ) và sóng phản xạ gương (π − θ)**, với pha lan truyền tại vị trí tàu; RAO nước hở, chưa xét ảnh hưởng của tường lên RAO hay cộng hưởng khe. Ở tàu cách tường 250 m, sóng phản xạ làm RMS lực sóng tăng khoảng √(1+Kr²): +5% với Kr=0,3 đến +36% với Kr=0,9. Không có `hydro_reference` thì dùng RAO minh họa trong cấu hình; các trục có tải sóng bằng 0 được ghi vào `manifest.json` (`wave_zero_axes`) và báo cáo. Các nhãn tải sạch được dùng làm lịch sử; chưa đánh giá nhiễu cảm biến tải.

Các mô hình LSTM điểm, Energy-LSTM và GAN-LSTM chạy tối đa 100 epoch (patience=100). Chuẩn hóa fit trên các dòng train không lặp; checkpoint chọn theo validation. Đầu ra ràng buộc cực đại >= RMS >= 0. Không dự báo dấu của lực hoặc lực tức thời. Tập calibration giữ riêng, chưa dùng vì chưa xác định ngưỡng tải; chưa xuất xác suất vượt ngưỡng hoặc nhãn an toàn.

Kết quả gồm `report.md`, `evaluation/load_metrics.csv` (SI: N và Nm), `evaluation/load_loss.png`, `evaluation/load_forecast.png`, `*/best.pt`, `*/history.csv` và manifest mô phỏng. Các biểu đồ đổi đơn vị sang kN/kNm. Đây là một seed trên cùng họ mô phỏng, không phải kiểm chứng thực địa.

Nạp checkpoint tải và chạy dự báo từ 60 dòng cuối một CSV:

```powershell
.venv/Scripts/python.exe -m berth_sim.training.force_predict --checkpoint outputs/force_training_v1/gan_lstm/best.pt --input outputs/force_training_v1/dataset/scenario_048.csv --output outputs/force_prediction
```

Mã huấn luyện ở `src/berth_sim/training/`. Cấu hình `configs/training.json` tạo 56 kịch bản độc lập, mỗi kịch bản 4 giờ. Chia toàn bộ kịch bản thành train/validation/calibration/test theo tỷ lệ 32/8/8/8, sau đó mới cắt cửa sổ. Các kịch bản khác nhau cả tham số vật lý lẫn seed. Nghiên cứu hiện chỉ kiểm tra trong cùng họ mô phỏng xung thời tiết; chưa kiểm tra chuyển miền ra dữ liệu thực.

Đầu vào là 60 dòng đặc trưng một phút: hai thành phần gió tuyệt đối, gió giật và Hm0 sóng phản xạ ước lượng từ cảm biến. Mô hình dự báo chuỗi 30 phút. Không dùng sóng phản xạ sạch, tham số kịch bản tương lai hoặc trạng thái tàu làm đầu vào.

Ba mô hình được huấn luyện:

- `lstm_point`: LSTM dự báo điểm, tối ưu MSE trong không gian vật lý đã chuẩn hóa.
- `energy_lstm`: LSTM encoder–decoder sinh ensemble với nhiễu chung cho toàn chuỗi; hai nhánh đầu ra gió/sóng; energy score cộng sai số trung bình ensemble.
- `gan_lstm`: cùng kiến trúc generator, thêm discriminator có điều kiện lịch sử và loss adversarial không bão hòa. Ba epoch đầu chưa thêm adversarial loss vào generator.

Generator dự báo phần dư quanh trạng thái cuối. Các biến dương được biểu diễn bằng log của phần tăng gió giật và log năng lượng sóng. Đầu ra vật lý thỏa Hm0 > 0 và gió giật >= độ lớn vector gió trung bình. Khi tính loss, giải mã về các biến vật lý rồi chuẩn hóa theo tập train; không đánh giá CRPS trên log năng lượng.

Cài môi trường GPU phù hợp với máy và chạy từ thư mục dự án. Cấu hình hiện tại dùng `"device": "cuda"`. Các lệnh dưới đây là ví dụ cho môi trường CUDA 13.0:

```powershell
uv --cache-dir .tmp/uv-cache pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv --cache-dir .tmp/uv-cache pip install --python .venv/Scripts/python.exe matplotlib
$env:PYTHONPATH = 'src'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tmp/matplotlib'
.venv/Scripts/python.exe -m berth_sim.training.run --config configs/training.json --output outputs/training_v2
```

Máy không có GPU: cài torch từ `https://download.pytorch.org/whl/cpu` và đặt `"device": "cpu"`. Nếu cấu hình yêu cầu `cuda` mà không có GPU, chương trình báo lỗi thay vì âm thầm chạy CPU. Trên GPU, code đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` để bật kernel tất định; kết quả GPU và CPU có thể khác nhau ở chữ số cuối nên cần so trong cùng thiết bị.

Sinh dữ liệu mô phỏng (NumPy) là bước chậm nhất và chạy trên CPU. `data_workers` (mặc định cấu hình: 8) chạy các kịch bản song song. Cấu hình kịch bản được rút tuần tự từ seed trước khi mô phỏng, nên dữ liệu trùng khớp tuyệt đối với khi chạy tuần tự (`data_workers: 1`).

Mỗi thí nghiệm cần thư mục mới. Có thể tái sử dụng thư mục dataset đã chuẩn bị với đúng cấu hình trước khi bắt đầu huấn luyện; không tự ghi đè hoặc tiếp tục một thí nghiệm đã có `experiment.json`. Checkpoint tốt nhất được chọn chỉ bằng validation; không chọn theo điểm test. Cấu hình hiện tại chạy 100 epoch với patience=100, nên không dừng sớm trong lần chạy này.

Trong thư mục thí nghiệm:

| Đường dẫn | Nội dung |
|---|---|
| `dataset/manifest.json` | Cấu hình đầy đủ của từng kịch bản, chia tập, số cửa sổ |
| `dataset/*.npz` | Dữ liệu cửa sổ riêng từng tập và ID kịch bản |
| `scaler.json` | Chuẩn hóa fit duy nhất trên tập train |
| `lstm_point/best.pt` | Checkpoint dự báo điểm |
| `energy_lstm/best.pt` | Checkpoint mô hình sinh không adversarial |
| `gan_lstm/best.pt` | Checkpoint conditional GAN–LSTM |
| `*/history.csv` | Loss huấn luyện và điểm validation mỗi epoch |
| `evaluation/forecast_metrics.csv` | MAE, RMSE, CRPS, coverage, độ rộng khoảng dự báo |
| `evaluation/warning_metrics.csv` | Brier thô/hiệu chỉnh tại 5, 10, 30 phút |
| `evaluation/calibrators.json` | Hiệu chỉnh logistic từ tập calibration; ghi rõ fallback nếu chỉ có một lớp |
| `evaluation/*_test.npz` | Ensemble test, quan trắc đích, ID kịch bản, thời điểm phát hành |
| `report.md` | Báo cáo kết quả và biểu đồ |
| `source/` | Bản chụp mã nguồn của thí nghiệm |

Các khoảng tin cậy được bootstrap theo toàn kịch bản, không coi cửa sổ chồng lấn là độc lập. Ngưỡng 13 m/s và 0.7 m là **ngưỡng thử nghiệm**. Xác suất được xét cho gió, sóng, đồng thời cùng bước thời gian, hoặc ít nhất một yếu tố trong chân trời. Chưa triển khai cảnh báo theo đợt/lead time. Hiệu chỉnh xác suất sự kiện không tự sửa coverage của khoảng dự báo từng biến.

Dự báo bằng checkpoint đã huấn luyện; CSV cần ít nhất 60 dòng liên tiếp hợp lệ:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m berth_sim.training.predict --checkpoint outputs/training_v1/gan_lstm/best.pt --features outputs/training_v1/dataset/scenario_048.csv --calibrators outputs/training_v1/evaluation/calibrators.json --output outputs/prediction_demo
```

Lệnh trên dùng 60 dòng cuối CSV để dự báo tiếp 30 phút; khác với biểu đồ test, vốn có sẵn tương lai quan sát để đối chiếu. Kết quả là `forecast.csv`, `forecast.npz`, `events.json`. Nạp checkpoint từ nguồn tin cậy; code dùng `weights_only=True`.

Kiểm thử toàn bộ mô phỏng và huấn luyện: `.venv/Scripts/python.exe -m unittest discover -s tests -v` với `PYTHONPATH=src`. Nếu chưa cài PyTorch, nhóm kiểm thử huấn luyện được bỏ qua.

## Thống kê kết quả cho bài báo

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe scripts/paper_results.py          # hình 300 dpi + outputs/paper_results/tables.json
.venv/Scripts/python.exe scripts/paper_3d.py               # hình 3D (phổ S_R(f,t) 8 kịch bản test, mặt biển + tàu), cần Playwright
.venv/Scripts/python.exe scripts/ablation_summary.py        # trung bình ± độ lệch chuẩn qua seed (v3 base ↔ nâng cấp)
cd scripts/paper_docx; npm install docx; node build.js ../../outputs/paper_results ../../Ket_qua_bai_bao_song_ngu_v3.docx
```

`Ket_qua_bai_bao_song_ngu_v3.docx`: hình (chữ Arial) và bảng tiếng Anh, nhận xét tiếng Việt; dữ liệu `outputs/v3_data` (128/16/16/24 kịch bản), ablation base ↔ nâng cấp 3 seed (`configs/training_v3_*.json`), hình chi tiết từ `v3_upgraded_s2` (seed trung vị); phần tải vẫn từ `force_training_v2`. Nhận xét lấy số trực tiếp từ `tables.json`/`ablation_v3.json`. Nhận xét trong `build.js` được viết theo số liệu lần chạy này; khi huấn luyện lại cần đọc lại và cập nhật các câu nhận định định tính.

## Nguồn mô hình

- Fossen, mô hình marine craft: https://www.fossen.biz/html/marineCraftModel.html
- Fossen và Perez (2009), tải môi trường và mô hình đáp ứng sóng: https://www.fossen.biz/publications/2009%20Fossen%20and%20Perez%20IEEE%20CST.pdf
- Perez và Fossen (2006), phân biệt chồng chất lực và chuyển động: https://fossen.biz/publications/2006%20Perez%20and%20Fossen%20MCMC.pdf
- MSS, biểu diễn trạng thái và sóng bất quy tắc: https://fossen.biz/publications/2005%20Smogeli%20et%20al%20IMAM.pdf
- Kaimal/TurbSim: https://www.nrel.gov/wind/nwtc/assets/downloads/TurbSim/TurbSim_v2.00.pdf
- Mansard và Funke, tách sóng tới/phản xạ: https://doi.org/10.9753/icce.v17.8
- Zelt và Skjelbreia (1992), tách sóng least squares với số đầu đo tùy ý, Proc. 23rd ICCE
- Welch (1967), ước lượng phổ trung bình đoạn: https://doi.org/10.1109/TAU.1967.1161901
- DNV-RP-C205, Environmental conditions and environmental loads, mục 3.5.5 (JONSWAP)
- Tucker, Challenor và Carter (1984), Numerical simulation of a random sea: a common error and its effect upon wave group statistics, Applied Ocean Research 6(2)
- Zanuttigh và van der Meer (2008), Wave reflection from coastal structures in design conditions, Coastal Engineering 55(10), 771–779; cũng trong EurOtop (2018)

Đây là triển khai mới bằng NumPy, không sao chép mã MSS. Phần phản xạ tuyến tính và kịch bản thời tiết là phần mở rộng được ghi rõ, không gán cho mô hình tàu Fossen.
