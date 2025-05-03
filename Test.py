import pandas as pd  # type: ignore

# Đọc file CSV đã upload
df = pd.read_csv('D:\VideoGestureControl\hand_gesture_dataset.csv')

# Xem tổng số dòng (số mẫu)
print("Tổng số mẫu:", len(df))

# Xem thống kê từng label (từng loại cử chỉ)
print("\nSố mẫu theo từng cử chỉ:")
print(df['label'].value_counts())