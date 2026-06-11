import zipfile
import pandas as pd
import os
from tqdm import tqdm

# Lấy đường dẫn tuyệt đối của file code (thư mục model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "archive.zip")

# Lùi ra 1 cấp để tạo thư mục dataset ở ngoài cùng (cùng cấp với thư mục model)
PROJECT_DIR = os.path.dirname(BASE_DIR)
SAVE_DIR = os.path.join(PROJECT_DIR, "dataset")

# Tạo sẵn vỏ thư mục
os.makedirs(os.path.join(SAVE_DIR, "male"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "female"), exist_ok=True)

def main():
    print(f"Đang phân tích bản đồ file Zip: {ZIP_PATH}...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # 1. Lấy toàn bộ danh sách file trong ZIP để đối chiếu
        all_zip_files = z.namelist()
        
        # 2. Tạo một từ điển map tên file gốc với đường dẫn thật trong ZIP
        file_map = {}
        csv_path = None
        for f in all_zip_files:
            base_name = os.path.basename(f)
            file_map[base_name] = f
            if base_name == "cv-valid-train.csv":
                csv_path = f
                
        if not csv_path:
            print("\n[LỖI CỰC ĐỘ] Không tìm thấy file cv-valid-train.csv trong Zip! Hãy kiểm tra lại file archive.zip")
            return
            
        print("1. Đang đọc cấu trúc CSV...")
        with z.open(csv_path) as csv_file:
            df = pd.read_csv(csv_file)
            
        df = df.dropna(subset=['gender'])
        male_df = df[df['gender'] == 'male'].head(5000)
        female_df = df[df['gender'] == 'female'].head(5000)
        
        def extract_audio(target_df, gender_label):
            print(f"\n2. Đang giải nén {len(target_df)} file {gender_label} (sẽ mất một chút thời gian)...")
            
            success_count = 0
            for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
                # Lấy tên file gốc (ví dụ: sample-000000.mp3)
                base_name = os.path.basename(row['filename'])
                
                # Tìm đường dẫn thực tế trong file Zip để giải nén
                if base_name in file_map:
                    real_zip_path = file_map[base_name]
                    audio_data = z.read(real_zip_path)
                    
                    with open(os.path.join(SAVE_DIR, gender_label, base_name), "wb") as out_f:
                        out_f.write(audio_data)
                    success_count += 1
                    
            print(f"-> HOÀN TẤT: Giải nén thành công {success_count}/{len(target_df)} file {gender_label}.")

        extract_audio(male_df, "male")
        extract_audio(female_df, "female")
        
    print(f"\n[XONG] Dữ liệu đã được tự động đặt đúng vị trí chuẩn: {SAVE_DIR}")

if __name__ == "__main__":
    main()