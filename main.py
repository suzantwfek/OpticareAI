from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI(title="Glaucoma Detection API")

# السماح بجميع الـ Origins لتجنب مشاكل الـ CORS أثناء التنقل بين الصفحات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل موديل الذكاء الاصطناعي المسجل باسم model.h5
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# لربط المجلد الرئيسي بالملفات الثابتة
app.mount("/static", StaticFiles(directory="."), name="static")

# 1. مسار الصفحة الرئيسية للويب سايت (Index)
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 2. مسار صفحة تسجيل الدخول (Login)
@app.get("/login.html", response_class=HTMLResponse)
def get_login_page():
    with open("login.html", "r", encoding="utf-8") as f:
        return f.read()

# 3. مسار صفحة لوحة التحكم (Dashboard)
@app.get("/dashboard.html", response_class=HTMLResponse)
def get_dashboard_page():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# 4. مسار صفحة الفحص ورفع الصور (Scan)
@app.get("/scan.html", response_class=HTMLResponse)
def get_scan_page():
    with open("scan.html", "r", encoding="utf-8") as f:
        return f.read()

# 5. مسار صفحة إظهار النتيجة النهائية (Results)
@app.get("/results.html", response_class=HTMLResponse)
def get_results_page():
    with open("results.html", "r", encoding="utf-8") as f:
        return f.read()

# 6. مسار صفحة الإعدادات (Settings) - تم إضافتها بناءً على طلبك
@app.get("/settings.html", response_class=HTMLResponse)
def get_settings_page():
    with open("settings.html", "r", encoding="utf-8") as f:
        return f.read()


# دالة التنبؤ وفحص الصور بالذكاء الاصطناعي المحدثة لحل مشكلة ثبات النتيجة
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(
            status_code=400,
            content={"error": "يرجى رفع صورة بصيغة JPG أو PNG فقط"}
        )

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        # عمل التنبؤ باستخدام الموديل وحفظ المخرجات الخام
        predictions = model.predict(img_array)
        raw = predictions[0]

        # طباعة المخرجات الرقمية في الـ Logs بتاعة Railway عشان نراقب الأرقام بدقة
        print(f"--- Model Raw Output: {raw} ---")

        # الاحتمال الأول: إذا كان الموديل بيطلع قيمة واحدة فقط بين الـ 0 والـ 1 (Binary Classification بناتج واحد)
        if len(raw) == 1 or (hasattr(raw, 'shape') and len(raw.shape) == 1 and raw.shape[0] == 1):
            prob_glaucoma = float(raw[0])
            
            # عتبة القرار (إذا كان الاحتمال أكبر من 50% فهو مريض، أقل فهو سليم)
            if prob_glaucoma >= 0.5:
                label = "Glaucoma"
                confidence = round(prob_glaucoma * 100, 2)
                is_glaucoma = True
            else:
                label = "Normal"
                confidence = round((1 - prob_glaucoma) * 100, 2)
                is_glaucoma = False
                
        # الاحتمال الثاني: إذا كان الموديل بيطلع قيمتين مصفوفة احتمالات
        else:
            predicted_index = int(np.argmax(raw))
            confidence = round(float(np.max(raw)) * 100, 2)
            
            # الترتيب القياسي (الفئة 0 تعني جلوكوما، الفئة 1 تعني طبيعي)
            if predicted_index == 0:
                label = "Glaucoma"
                is_glaucoma = True
            else:
                label = "Normal"
                is_glaucoma = False

        return {
            "prediction": label,
            "confidence": confidence,
            "is_glaucoma": is_glaucoma
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"حدث خطأ أثناء التحليل: {str(e)}"}
        )