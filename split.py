import pandas as pd
import os

# Đường dẫn
metadata_path = "data/raw/ham10000/HAM10000_metadata.csv"
train_isic_path = "data/processed/isic2018_binary/isic_2018_binary_train.csv"
output_path = "data/processed/ham10000/ham10000_test_clean.csv"

# 1. Đọc metadata HAM10000
df_ham = pd.read_csv(metadata_path)

# 2. Quy đổi nhãn nhị phân dựa trên các cột hiện có
# Ác tính (Label 1) nếu một trong các cột MEL, BCC, hoặc AKIEC có giá trị 1.0
malignant_cols = ['MEL', 'BCC', 'AKIEC']

# Tạo cột label: nếu tổng của 3 cột này > 0 thì là 1, ngược lại là 0
df_ham['label'] = df_ham[malignant_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Chỉnh lại tên cột image cho khớp với ISIC (nếu cần)
# Trong ảnh thấy cột đầu tiên tên là 'image', ta đổi nó thành 'image_id' để so khớp với tập Train
df_ham = df_ham.rename(columns={'image': 'image_id'})

print("✅ Đã chuyển đổi nhãn thành công!")
print(df_ham['label'].value_counts())
# 3. Đọc danh sách ảnh đã dùng để Train ISIC để loại bỏ
df_train_isic = pd.read_csv(train_isic_path)

# Kiểm tra xem cột tên ảnh là 'image' hay 'image_id'
train_col = 'image' if 'image' in df_train_isic.columns else 'image_id'
train_ids = set(df_train_isic[train_col].values)

print(f"🔍 Đã tìm thấy {len(train_ids)} ảnh trong tập Train ISIC (Cột: {train_col})")
# 4. Lọc: Chỉ giữ lại những ảnh KHÔNG nằm trong tập Train
df_ham_clean = df_ham[~df_ham['image_id'].isin(train_ids)]

print(f"Tổng số ảnh HAM10000: {len(df_ham)}")
print(f"Số ảnh sạch dùng để Test (không trùng tập Train): {len(df_ham_clean)}")

# 5. Lưu file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_ham_clean[['image_id', 'label']].to_csv(output_path, index=False)