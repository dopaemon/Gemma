# Plan

Kế hoạch cho adapter RE. Cập nhật khi có kết quả đo, không phải khi có ý tưởng.

## Giai đoạn 1 — train (đang chạy)

Adapter QLoRA rank 64 / 42 layer trên 71.463 mẫu, một epoch = 35.731 iter.
Chi tiết tham số và lý do nằm trong README.md.

| | |
|---|---|
| Bắt đầu | 2026-09-24 02:43 |
| Tốc độ thật | ~0,14 it/s |
| Val loss | 2,282 (baseline) → 0,739 (iter 2.000) |
| Còn lại | ~2 ngày 20 tiếng tính từ 2026-09-24 12:30 |

Đã gián đoạn một lần lúc 12:01 vì lỗi Metal `ImpactingInteractivity`
(macOS giết lệnh GPU để giữ giao diện mượt). Watchdog bắt được, không mất
trọng số.

**Xong khi nào:** log xuất hiện `Saved final weights`. Watchdog tự thoát lúc đó.

## Giai đoạn 2 — đo trước, quyết sau

Chưa có cách đo nào ngoài val loss, mà val loss thấp **không** đảm bảo tên đặt
ra hữu ích. Phải dựng bài kiểm tra trước khi quyết bất cứ điều gì về dữ liệu
tiếp theo.

Bài kiểm tra tối thiểu:

- 200 mẫu rút từ `re_data_all/valid.jsonl` (model chưa từng thấy khi train)
- Cho model đặt tên, so với nhãn thật
- Đếm ba con số: **trùng chính xác**, **gần đúng về nghĩa**, **sai hẳn**
- Chạy cùng bộ đó trên model gốc không gắn adapter để có mốc so sánh

Không có mốc so sánh thì con số tuyệt đối vô nghĩa — 40% trùng chính xác là
tốt hay tệ phụ thuộc hoàn toàn vào model gốc được bao nhiêu.

Thêm một bài thủ công: lấy một binary thật của mình, `-O2`, strip, decompile
bằng Ghidra rồi đưa cho model. Đây mới là điều kiện sử dụng thật, khác hẳn
`-O0` trong corpus.

## Giai đoạn 3 — vòng dữ liệu tiếp theo

Chỉ bắt đầu sau khi có số từ giai đoạn 2.

**Mặc định là train tiếp, không train lại.** Nạp adapter cũ rồi train trên
corpus mới. Train lại từ đầu chỉ khi:

1. Đổi `rank`, `num_layers`, hoặc `max_seq_length` — trọng số cũ không khớp
   hình dạng nữa.
2. Phát hiện corpus hiện tại có lỗi hệ thống (như vụ atul10 rò đáp án 39–55%).

**Bắt buộc trộn lại khi thêm task mới.** Corpus mới phải kèm ~30% mẫu rút từ
`re_data_all` hiện tại, nếu không model quên kỹ năng cũ trong vài nghìn bước.

**Khi chốt vòng cuối**, sửa `lr_schedule` cho LR giảm thật về 3e-6. Hiện mỗi
lần resume là warmup lại từ đầu nên chặng giảm chưa bao giờ xảy ra.

### Hướng mở rộng, xếp theo thứ tự tôi nghĩ đáng làm

Chưa cam kết cái nào — chờ số đo.

1. **Binary `-O2`/`-O3`** — khoảng trống lớn nhất. Corpus hiện tại toàn `-O0`,
   còn binary release thật đều tối ưu hoá: vòng lặp unroll, hàm inline, trông
   rất khác. Khó ở chỗ phải tìm bộ `-O2` không rò đáp án; các bộ atul10 đã bị
   loại vì lý do này.
2. **Assembly thô** — hiện bắt buộc phải decompile bằng Ghidra trước. Dạy đọc
   asm trực tiếp sẽ bỏ được bước đó.
3. **Tìm lỗ hổng** — bigvul, devign, juliet đã khảo sát và tạm loại. Đây là
   task khác hẳn, cần lượng dữ liệu lớn mới có tác dụng thật.
4. **Gỡ obfuscation** — ollvm. Hẹp, để sau cùng.

## Những gì đã loại và vì sao

Ghi lại để khỏi khảo sát lại lần nữa.

| Bộ | Lý do loại |
|---|---|
| `atul10/*_O2_*` | 39% (x86) và 55% (arm) số dòng có đáp án nằm sẵn trong prompt |
| `atul10/*_O0_*` | nhãn lấy từ cột `clean_raw_generation`, là model khác sinh ra chứ không phải nhãn thật |
| `atul10/prompt_obfuscated_binaries_orig` | `stripped_function_name` trùng y hệt tên thật |
| `cybersecserver/Ghidra-Instruct-10K` | trùng byte-for-byte với bản gốc `RevEng-24-25`, lại có mẫu ghép sai |
| `bstee615/bigvul`, `DetectVul/devign`, `LorenzH/juliet` | task tìm lỗ hổng, để dành vòng sau |
| `oleksiihrush/ollvm-decompilation-dataset` | task gỡ obfuscation, để dành vòng sau |
