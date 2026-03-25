# Hướng dẫn train và chạy project

## 1. Chuẩn bị môi trường

### Bước 1: Mở terminal tại thư mục project

```powershell
cd C:\SV2026
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Bước 3: Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

Hoặc cài từng gói:

```powershell
pip install torch torchvision torchsummary ptflops
```

---

## 2. Train mô hình

Script train chính: **`test_CIFAR10.py`**.

### Chạy nhanh (mặc định: CPU, CIFAR-10, tự tải dataset)

```powershell
python test_CIFAR10.py --download
```

- Dataset CIFAR-10 sẽ tải về thư mục `../../dataML` (hoặc thư mục bạn chỉ định với `-r`).
- Mặc định dùng **CPU** (`-g -1`). Train 200 epoch có thể rất lâu trên CPU.

### Chỉ định thư mục chứa dataset

```powershell
python test_CIFAR10.py -r D:\datasets --download
```

### Dùng GPU (nếu có NVIDIA + CUDA)

```powershell
python test_CIFAR10.py -g 0 --download
```

### Train mạng DualBranchNet (mạng dual-branch 32×32)

Trong file `test_CIFAR10.py`, tìm dòng:

```python
model = TestNet()
```

Đổi thành:

```python
model = DualBranchNet()
```

Rồi chạy:

```powershell
python test_CIFAR10.py --download
```

### Một số tham số hữu ích

| Tham số | Mặc định | Ý nghĩa |
|--------|----------|--------|
| `-r`, `--data-root` | `../../dataML` | Thư mục chứa dataset |
| `-d`, `--dataset` | `cifar10` | Dataset: `cifar10`, `cifar100`, `dogs` |
| `--download` | Tắt | Bật để tự tải dataset |
| `-g`, `--gpu-id` | `-1` (CPU) | GPU id (0, 1, ...) hoặc -1 cho CPU |
| `-b`, `--batch-size` | 128 | Batch size (giảm nếu hết RAM, ví dụ 32) |
| `-e`, `--epochs` | 200 | Số epoch |
| `-l`, `--learning-rate` | 0.1 | Learning rate |
| `-j`, `--workers` | 4 | Số worker load data |

**Ví dụ:** Train 50 epoch, batch 64, CPU, tự tải CIFAR-10:

```powershell
python test_CIFAR10.py --download -e 50 -b 64
```

**Kết quả train:**

- Checkpoint (model tốt nhất theo val accuracy) lưu tại:  
  `./checkpoints/CIFAR10_MDFNet/checkpoint_epochXXXX_XX.XX.pth`
- Log accuracy từng epoch:  
  `./checkpoints/CIFAR10_MDFNet/.txt`

---

## 3. Xem cấu trúc model và số tham số

```powershell
python checkmodel.py
```

- In ra kiến trúc và số tham số.
- Mặc định dùng **TestNet** và input 224×224. Nếu dùng **DualBranchNet** (input 32×32), cần sửa trong `checkmodel.py`: import `DualBranchNet`, dùng `DualBranchNet()` và `summary(..., (3, 32, 32), device='cpu')`.

---

## 4. Đo FLOPs / số tham số (ptflops)

```powershell
python ptflops_count.py
```

- In MACs, FLOPs và số tham số.
- Hiện tại dùng **TestNet** và input 224×224. Để đo **DualBranchNet** (32×32), sửa trong `ptflops_count.py`: dùng `DualBranchNet()` và `(3, 32, 32)`.

---

## 5. Tóm tắt lệnh thường dùng

```powershell
# Vào thư mục project
cd C:\SV2026

# Kích hoạt venv (nếu dùng)
.\.venv\Scripts\activate

# Train CIFAR-10 (tự tải data, CPU)
python test_CIFAR10.py --download

# Train với GPU 0
python test_CIFAR10.py -g 0 --download

# Train nhanh thử (ít epoch, batch nhỏ)
python test_CIFAR10.py --download -e 5 -b 32

# Xem model
python checkmodel.py

# Đo FLOPs
python ptflops_count.py
```

---

## 6. Lưu ý

- **CPU:** Nên giảm `-b` (ví dụ 32 hoặc 64) nếu máy chậm hoặc hết RAM.
- **Dataset:** Lần đầu nên dùng `--download`; sau đó có thể bỏ và dùng `-r` trỏ đúng thư mục đã tải.
- **DualBranchNet** thiết kế cho ảnh **32×32** (CIFAR-10); **TestNet** trong script gốc dùng với 224×224 trong checkmodel/ptflops.
