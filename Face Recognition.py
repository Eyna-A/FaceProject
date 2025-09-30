import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# --- 1. تنظیمات اولیه و احراز هویت Kaggle (اگر نیازی به دانلود دیتاست ندارید، این بخش را حذف کنید) ---
# این بخش برای اطمینان از دسترسی به API Kaggle است.
# فرض بر این است که فایل kaggle.json در مسیر C:\Users\msi\Desktop وجود دارد.
# try:
#     os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\msi\Desktop'
#     from kaggle.api.kaggle_api_extended import KaggleApi
#     api = KaggleApi()
#     api.authenticate()  
#     print("Kaggle API token loaded successfully!")
# except Exception as e:
#     print(f"Warning: Could not initialize Kaggle API. Error: {e}")

# --- 2. تعریف ابزارها ---
detector = MTCNN()
embedder = FaceNet()
DATABASE_FOLDER = 'images_database' # پوشه حاوی عکس‌های آموزشی شما

# --- 3. توابع کمکی ---

def extract_face_rgb(image_rgb, required_size=(160,160)):
    """تشخیص و استخراج چهره اول و بازگرداندن مختصات جعبه مرزی آن."""
    res = detector.detect_faces(image_rgb)
    if not res: 
        # اگر چهره‌ای پیدا نشد، None و مختصات خالی را برمی‌گرداند
        return None, None
    # گرفتن مختصات اولین چهره پیدا شده
    x, y, w, h = res[0]['box']
    x, y = max(0,x), max(0,y) # جلوگیری از مختصات منفی
    
    # استخراج و تغییر اندازه چهره
    face = image_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    
    # برگرداندن تصویر چهره و مختصات جعبه مرزی (bounding box)
    return face, (x, y, w, h)

def build_database(images_folder):
    """ایجاد پایگاه داده از ایمبِدینگ‌های چهره‌های موجود در پوشه."""
    embeddings = []
    labels = []
    print(f"Building database from {images_folder}...")
    
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Database folder not found: {images_folder}")
        
    for f in os.listdir(images_folder):
        img_path = os.path.join(images_folder, f)
        img = cv2.imread(img_path)
        
        if img is None: 
            print(f"Skipping non-image file: {f}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # از تابع جدید که دو خروجی دارد استفاده می‌کنیم، اما خروجی دوم (bbox) را نادیده می‌گیریم
        face, _ = extract_face_rgb(img_rgb)
        
        if face is not None:
            emb = embedder.embeddings([face])[0]
            emb = emb / np.linalg.norm(emb)  # نرمال‌سازی (Normalization)
            embeddings.append(emb)
            labels.append(os.path.splitext(f)[0])  # نام فایل به عنوان لیبل
            print(f"Face extracted and embedded for: {f}")
        else:
             print(f"No face detected in file: {f}")

    if not embeddings:
        raise ValueError("No faces found in database! Check your image files.")
        
    return np.vstack(embeddings), labels

# --- 4. منطق اصلی برنامه (Main Logic) ---

def initialize_webcam():
    """تلاش برای باز کردن دوربین با اندیس‌های مختلف (0، 1 و 2) با استفاده از بک‌اند DSHOW."""
    global cap 
    cap = None
    
    # تلاش برای باز کردن دوربین با اندیس‌های متداول
    # CAP_DSHOW استفاده از DirectShow API ویندوز را اجباری می‌کند
    for index in range(3):
        print(f"Attempting to open webcam with index: {index} using CAP_DSHOW...")
        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # <-- تغییر اصلی اینجا اعمال شد
        
        # Check if webcam opened successfully
        if temp_cap.isOpened():
            cap = temp_cap
            # تنظیم کیفیت فریم (اختیاری، اما می‌تواند در رفع مشکل کمک کند)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Webcam opened successfully with index {index}.")
            return cap
        # If not opened, release the temporary object
        temp_cap.release()

    return None

try:
    # 1. ساخت پایگاه داده
    db_embeddings, db_labels = build_database(DATABASE_FOLDER)
    
    # 2. راه‌اندازی دوربین با امتحان اندیس‌های مختلف
    cap = initialize_webcam()

    if cap is None:
        raise IOError("Cannot open webcam. Please check connections or ensure no other application is using the camera.")

    print("Webcam started. Press 'q' to quit.")
    
    # 3. حلقه اصلی تشخیص چهره
    while True:
        # متد read() استاندارد را برگرداندیم چون grab/retrieve ممکن است در برخی بک‌اندها مشکل داشته باشد.
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame (camera disconnected or busy).")
            break

        # الف) تبدیل به RGB برای استفاده با MTCNN و FaceNet
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # دریافت تصویر چهره و مختصات جعبه مرزی
        face, bbox = extract_face_rgb(rgb)

        if face is None:
            cv2.putText(frame, "No Face Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        else:
            # ب) محاسبه ایمبِدینگ چهره زنده و نرمال‌سازی
            emb = embedder.embeddings([face])[0]
            emb = emb / np.linalg.norm(emb)

            # ج) محاسبه فاصله کسینوسی (Cosine Distance) تا پایگاه داده
            scores = [cosine(e, emb) for e in db_embeddings]
            min_score = min(scores)
            min_index = scores.index(min_score)

            # د) تعیین آستانه (Threshold) و نتیجه
            threshold = 0.5  # آستانه را می‌توانید تغییر دهید
            
            # پیدا کردن نزدیک‌ترین شخص
            if min_score < threshold:
                name = db_labels[min_index]
                text = f"Access Granted: {name} ({min_score:.2f})"
                color = (0,255,0) # سبز
            else:
                text = f"Access Denied ({min_score:.2f})"
                color = (0,0,255) # قرمز
                
            # کشیدن جعبه مرزی دور چهره
            x, y, w, h = bbox
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            
            # نمایش متن در نزدیکی جعبه مرزی
            cv2.putText(frame, text, (x, y-10 if y-10>10 else y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        # ه) نمایش فریم نهایی
        cv2.imshow("Face Lock", frame)

        # و) شرط خروج از حلقه
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}. Please create the '{DATABASE_FOLDER}' folder and place images inside.")
except ValueError as e:
    print(f"FATAL ERROR: {e}. Please ensure you have valid face images in the database folder.")
except IOError as e:
    print(f"FATAL ERROR: {e}. Check if your webcam is connected and not being used by another application.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
finally:
    # 5. آزادسازی منابع در هر صورت
    # بررسی می‌کنیم که آیا cap تعریف شده است
    if 'cap' in locals() and cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")