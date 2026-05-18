from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles # استيراد عشان الملفات تشتغل
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

app = FastAPI(title="Glaucoma Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- [تعديل مهم جداً] ---
# السطر ده بيخلي السيرفر يشوف كل صفحات الـ HTML والملفات اللي جنبه في الفولدر
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    # هنا بنقول للسيرفر أول ما اللينك يفتح، اعرض صفحة الـ login أو index الرئيسية بتاعتك
    # اتأكدي من اسم الملف هنا (مثلاً لو البداية login.html أو index.html)
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

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

        predictions = model.predict(img_array)
        raw = predictions[0]

        print(f"DEBUG - output shape: {raw.shape}")
        print(f"DEBUG - raw output: {raw}")

        if raw.shape[0] == 1:
            # --- [تصليح حسابات السيجويد المقلوبة] ---
            prob_glaucoma = float(raw[0])
            if prob_glaucoma >= 0.5: # القيمة الكبيرة تعني وجود المرض
                label = "Glaucoma"
                confidence = round(prob_glaucoma * 100, 2)
                is_glaucoma = True
            else:
                label = "Normal"
                confidence = round((1 - prob_glaucoma) * 100, 2)
                is_glaucoma = False

        else:
            # softmax - قيمتين (دي مظبوطة)
            predicted_index = int(np.argmax(raw))
            confidence = round(float(np.max(raw)) * 100, 2)
            label = "Glaucoma" if predicted_index == 0 else "Normal"
            is_glaucoma = predicted_index == 0

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