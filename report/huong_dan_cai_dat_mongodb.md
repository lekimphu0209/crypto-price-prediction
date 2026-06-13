# Hướng dẫn cài đặt MongoDB trên Windows

## Bước 1: Tải MongoDB Server

1. Truy cập: https://www.mongodb.com/try/download/community
2. Chọn:
   - Version: 6.0 hoặc 7.0 (newest)
   - Platform: Windows
   - Package: msi
3. Click Download

## Bước 2: Cài đặt MongoDB Server

1. Chạy file .msi vừa tải
2. Chọn "Complete" installation
3. **Quan trọng:** Tích chọn "Install MongoDB as a Service"
4. Tích chọn "Install MongoDB Compass" (GUI tool)
5. Data Directory: C:\data\db (mặc định)
6. Click Install

## Bước 3: Cấu hình MongoDB

### Cách 1: Dùng MongoDB Compass (GUI)
1. Mở MongoDB Compass
2. Connection string: `mongodb://localhost:27017/`
3. Click Connect

### Cách 2: Dùng Command Line
```powershell
# Mở PowerShell as Administrator
# Kiểm tra MongoDB service
Get-Service MongoDB

# Nếu service không chạy, start nó
Start-Service MongoDB

# Test kết nối
mongo
```

## Bước 4: Tạo Database cho dự án

### Cách 1: Dùng MongoDB Compass
1. Connect tới MongoDB
2. Click "Create Database"
3. Database name: `crypto_prediction`
4. Collection name: `predictions` (hoặc bất kỳ)

### Cách 2: Dùng Command Line
```powershell
# Connect to MongoDB
mongo

# Tạo database và collection
use crypto_prediction
db.createCollection("predictions")
db.predictions.insertOne({test: "data"})
```

## Bước 5: Cấu hình dự án

1. Copy `.env.example` thành `.env`
2. Sửa `.env` file:
   ```
   DB_TYPE=mongo
   MONGO_CONNECTION_STRING=mongodb://localhost:27017/
   MONGO_DATABASE=crypto_prediction
   ```

## Bước 6: Install Python Driver

```powershell
# Install pymongo
pip install pymongo

# Hoặc install từ requirements.txt
pip install -r requirements.txt
```

## Bước 7: Test kết nối

Tạo file `test_mongo.py`:
```python
from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_prediction"]
    db.test_collection.insert_one({"test": "connection"})
    print("✅ MongoDB connection successful!")
    client.close()
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
```

Chạy: `python test_mongo.py`

## Troubleshooting

### MongoDB không start
```powershell
# Kiểm tra service
Get-Service MongoDB

# Start service
Start-Service MongoDB

# Nếu lỗi, cài đặt lại MongoDB
```

### Port 27017 bị chặn
```powershell
# Kiểm tra port
netstat -ano | findstr 27017

# Mở port trong Windows Firewall
# Settings > Windows Security > Firewall > Allow an app
```

### Connection refused
- Đảm bảo MongoDB service đang chạy
- Kiểm tra connection string trong `.env`
- Kiểm tra firewall/antivirus

## Docker Alternative (nếu không muốn cài trực tiếp)

```powershell
# Pull MongoDB image
docker pull mongo

# Run MongoDB container
docker run -d -p 27017:27017 --name mongodb mongo

# Connection string: mongodb://localhost:27017/
```

## Xóa MongoDB (nếu cần)

```powershell
# Stop service
Stop-Service MongoDB

# Uninstall từ Control Panel
# Xóa thư mục C:\Program Files\MongoDB
# Xóa thư mục C:\data\db
```

## Tài liệu tham khảo
- MongoDB Documentation: https://docs.mongodb.com/
- PyMongo Documentation: https://pymongo.readthedocs.io/
