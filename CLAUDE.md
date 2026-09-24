# CLAUDE.md

Quy tắc và ghi chú cho repo này. Đọc trước khi đụng vào train.

Kế hoạch nằm ở `PLAN.md`. Lý do đằng sau từng tham số nằm ở `README.md`.

## Đang chạy cái gì

Khi train hoạt động sẽ có ba tiến trình:

| tiến trình | việc | chết thì sao |
|---|---|---|
| `mlx_lm.lora` | train | watchdog khởi động lại trong 60 giây |
| `watchdog.sh` | canh train, tối đa 10 lần restart | train tiếp tục nhưng không còn ai canh |
| `caffeinate -s -w <pid>` | chặn máy ngủ, tự thoát khi train xong | máy có thể ngủ, train bị treo (không chết) |

Kiểm tra nhanh:

```bash
pgrep -fl "mlx_lm.lora|watchdog.sh|caffeinate -s"
```

## Quy tắc

**Dừng watchdog TRƯỚC khi dừng train.** Ngược lại thì nó khởi động lại ngay
đúng cái vừa giết.

```bash
kill $(pgrep -f watchdog.sh)
kill -TERM $(pgrep -f mlx_lm.lora)
```

**SIGINT không giết được mlx_lm.** Đã thử gửi liên tục hơn 2 phút, tiến trình
vẫn sống. Phải dùng `-TERM`. Ctrl+C trong terminal cũng không đáng tin.

**Dừng train là mất tối đa 500 iter.** mlx_lm chỉ lưu đúng mỗi mốc
`save_every`, không lưu khi bị ngắt. Muốn không mất gì thì đợi qua mốc rồi
mới dừng.

**Không chạy inference trong lúc train.** Hai tiến trình tranh GPU là nguyên
nhân của lỗi Metal đã gặp. Muốn thử adapter thì dừng train, hoặc copy
checkpoint ra chỗ khác rồi đợi train xong.

**Đọc log phải qua `tr '\r' '\n'`.** Thanh tiến trình tqdm ghi bằng `\r` nên
`tail` trả về một dòng khổng lồ:

```bash
tr '\r' '\n' < chain_logs/train_all.log | grep -E "^Iter" | tail -5
```

**`rtk` bóp méo văn bản dài.** Nó cắt bớt từ khi tóm tắt output. Khi cần xem
chính xác nội dung dataset hoặc chuỗi, dùng `rtk proxy` hoặc ghi ra file rồi
đọc file.

## Bẫy đã dẫm phải

**Bộ đếm iter reset về 1 mỗi lần resume.** mlx_lm chỉ nạp trọng số adapter,
không nạp optimizer state, không nhớ đã đi tới đâu:

```python
# lora.py:248-250
if args.resume_adapter_file is not None:
    model.load_weights(args.resume_adapter_file, strict=False)
```

Hệ quả:

- Sau một lần resume, file `0000500` chứa **nhiều** train hơn `0002000`. Vì
  vậy `train_re.sh` chọn checkpoint theo **thời gian sửa file**, không theo số
  thứ tự. Đã có lần chọn sai làm mất 500 iter.
- Lịch LR quay về warmup từ đầu. Resume nhiều lần thì chặng cosine giảm về
  3e-6 không bao giờ xảy ra. Vòng cuối phải sửa `lr_schedule` thủ công.
- Con số "còn bao nhiêu iter" theo log là của lượt chạy hiện tại, không phải
  tổng tiến độ.

**`seed: 0` trong config không có tác dụng.** `trainer.py:138` viết
`if seed:`, mà `0` là falsy. Thêm nữa `train()` gọi `iterate_batches()` mà
không truyền seed. Nên thứ tự xáo dữ liệu ngẫu nhiên mỗi lần chạy, và val
loss có nhiễu **±0,05** vì mỗi lần bốc 100 batch khác nhau. Chênh lệch dưới
mức đó đừng đọc thành tiến bộ.

**`RuntimeError: [METAL] ... ImpactingInteractivity`** là macOS giết lệnh GPU
để giữ giao diện mượt, không phải lỗi code hay hết RAM. Nhất thời, restart là
hết. Đã gặp một lần sau 5 giờ chạy.

**Mảng rỗng dưới `set -u` làm bash 3.2 báo lỗi.** macOS vẫn dùng bash 3.2.
Phải viết `${ARR[@]+"${ARR[@]}"}`.

## Quy tắc về dữ liệu

**Luôn kiểm tra rò đáp án trước khi nhận một dataset.** Rất nhiều bộ RE trên
HuggingFace khai là "stripped" nhưng vẫn giữ nguyên chữ ký hàm Ghidra khôi
phục được, nên đáp án nằm sẵn trong prompt. Những dòng đó dạy model chép chứ
không dạy suy luận, và chép là đúng cái sẽ hỏng trên binary strip thật.
`mix_data.py` có sẵn hàm `leaks()`.

Chỉ kiểm tra rò với task **đặt tên**. Với task dịch (decompile), đáp án
*đương nhiên* dùng lại chữ trong đầu vào — bật kiểm tra ở đó là xoá nhầm gần
hết dữ liệu.

**Kiểm tra nhãn đến từ đâu.** Cột tên `clean_raw_generation`,
`model_generated_*` nghĩa là model khác sinh ra, không phải nhãn thật. Đã loại
cả bộ atul10 vì lý do này.

**Cắt mẫu dài hơn `max_seq_length` thay vì để mlx_lm cắt.** Nó giữ N token
**đầu**, tức vứt mất đáp án ở cuối.

**`mask_prompt: true` là bắt buộc.** Không có nó, ~97% gradient đổ vào việc
học thuộc lòng code decompile thay vì học đặt tên.

## Về adapter

Rank 64 đụng 154,7M tham số nên **chat thường sẽ kém đi**. Dùng RE thì gắn
adapter, việc khác thì bỏ `ADAPTER_PATH` đi:

```bash
API_KEY=secret ADAPTER_PATH=./adapters uv run api.py   # chế độ RE
API_KEY=secret uv run api.py                           # chế độ thường
```

Prompt phải khớp định dạng lúc train, xem `prep_new.py` để lấy đúng câu chữ.
Sai định dạng thì kết quả tệ hơn nhiều.

## Git

Commit có `-s` (sign-off). Ba commit cũ `c2d91d8`, `5c7f11c`, `565a31f` thiếu
sign-off; sửa phải force-push nên đang để nguyên.

`nd_*/`, `re_data_all/`, `adapters/`, `gemma-mlx-4bit/` đều gitignore. Tạo lại
bằng `prep_new.py` và `mix_data.py`.
