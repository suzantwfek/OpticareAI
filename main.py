from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf
import io

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

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/login", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
def get_login_page():
    with open("login.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/register", response_class=HTMLResponse)
@app.get("/register.html", response_class=HTMLResponse)
def get_register_page():
    with open("register.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard.html", response_class=HTMLResponse)
def get_dashboard_page():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/scan", response_class=HTMLResponse)
@app.get("/scan.html", response_class=HTMLResponse)
def get_scan_page():
    with open("scan.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/results", response_class=HTMLResponse)
@app.get("/results.html", response_class=HTMLResponse)
def get_results_page():
    with open("results.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/history", response_class=HTMLResponse)
@app.get("/history.html", response_class=HTMLResponse)
def get_history_page():
    with open("history.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/settings", response_class=HTMLResponse)
@app.get("/settings.html", response_class=HTMLResponse)
def get_settings_page():
    with open("settings.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a JPG or PNG image only."}
        )

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        predictions = model.predict(img_array)
        raw = predictions[0]

        print(f"--- Model Raw Output: {raw} ---")

        # الموديل بيطلع قيمة واحدة بين 0 و 1
        # 0 = Glaucoma  |  1 = Normal
        if len(raw) == 1 or (hasattr(raw, 'shape') and raw.shape[0] == 1):
            prob = float(raw[0])

            # ✅ التعديل: القيمة القريبة من 0 تعني Glaucoma
            if prob < 0.5:
                label = "Glaucoma"
                confidence = round((1 - prob) * 100, 2)
                is_glaucoma = True
            else:
                label = "Normal"
                confidence = round(prob * 100, 2)
                is_glaucoma = False

        # الموديل بيطلع قيمتين [normal_prob, glaucoma_prob]
        else:
            predicted_index = int(np.argmax(raw))
            confidence = round(float(np.max(raw)) * 100, 2)

            # index 0 = Normal | index 1 = Glaucoma
            if predicted_index == 1:
                label = "Glaucoma"
                is_glaucoma = True
            else:
                label = "Normal"
                is_glaucoma = False

        print(f"--- Result: {label} | Confidence: {confidence}% | is_glaucoma: {is_glaucoma} ---")

        return {
            "prediction": label,
            "confidence": confidence,
            "is_glaucoma": is_glaucoma
        }

    except Exception as e:
        print(f"--- ERROR: {str(e)} ---")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during analysis: {str(e)}"}
        )