# exploration/

Thư mục này chứa các script **nghiên cứu / EDA (Exploratory Data Analysis)** độc lập.
Mục đích: khảo sát dữ liệu và tín hiệu **trước khi** đưa vào model chính.

## Scripts hiện có

| File | Mô tả |
|------|-------|
| `macro_correlation_analysis.py` | Phân tích correlation giữa Gold/DXY/Yield returns và BTC direction target |

## Nguyên tắc

- **Không import** từ `src/training/` hay các pipeline model chính
- Chỉ dùng raw data + pandas / scipy / matplotlib
- Kết quả ở đây là **tham khảo** để quyết định feature engineering, không phải kết quả cuối
