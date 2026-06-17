from models import ClientData, PredictionResponse, CSVPredictionResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import io
import json
import os
import logging
from sklearn.metrics import roc_auc_score
import uvicorn
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Mortgage Prediction Service",
    description="Сервис для предсказания одобрения ипотеки",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для pipeline
pipeline = None
feature_config = None


def load_feature_config():
    global feature_config
    config_path = "artifacts/feature_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            feature_config = json.load(f)
        logger.info(f"Конфигурация загружена")
    else:
        logger.warning("feature_config.json не найден")
        feature_config = {}


load_feature_config()


@app.post("/upload-model", summary="Загрузка обученной модели", tags=["Model"])
async def upload_model(file: UploadFile = File(...)):
    global pipeline

    try:
        if not file.filename.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="Файл должен быть .pkl")

        contents = await file.read()
        pipeline = joblib.load(io.BytesIO(contents))

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model.pkl", 'wb') as f:
            f.write(contents)

        logger.info(f"Pipeline загружен: {file.filename}")

        return {
            "status": "success",
            "message": "Pipeline успешно загружен",
            "filename": file.filename,
            "size": len(contents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/predict", summary="Получение предсказания", tags=["Prediction"])
async def predict(data: ClientData):
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Используйте POST /upload-model")

    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        logger.info(f"Получены данные: {input_dict}")

        # Получаем вероятность вместо предсказания
        proba = pipeline.predict_proba(df)[0]
        probability = float(proba[1] if len(proba) > 1 else proba[0])

        threshold = 0.2
        prediction = 1 if probability >= threshold else 0

        logger.info(f"Вероятность одобрения: {probability:.4f}, порог: {threshold}, предсказание: {prediction}")

        return {
            "loan_status": int(prediction),
            "loan_status_text": "Одобрено" if prediction == 1 else "Отказ",
            "confidence": probability,
            "input_data": input_dict
        }

    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")



@app.post("/predict-from-csv", summary="Предсказание из CSV", tags=["Prediction"])
async def predict_from_csv(
    file: UploadFile = File(...),
    return_full_dataset: bool = Query(False, description="Вернуть полный датасет с предсказаниями")
):
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")

    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть .csv")

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        original_df = df.copy()
        has_target = 'loan_status' in df.columns

        logger.info(f"Получен CSV: {file.filename}, строк: {len(df)}")

        if has_target:
            X = df.drop(columns=['loan_status'])
        else:
            X = df

        # Используем predict_proba для CSV тоже
        probabilities = pipeline.predict_proba(X)[:, 1]
        threshold = 0.2
        predictions = [1 if prob >= threshold else 0 for prob in probabilities]
        df['predicted_loan_status'] = predictions

        roc_auc = None
        if has_target:
            try:
                y_true = original_df['loan_status'].astype(int)
                y_pred = df['predicted_loan_status'].astype(int)
                if len(y_true.unique()) > 1:
                    roc_auc = float(roc_auc_score(y_true, y_pred))
            except Exception as e:
                logger.warning(f"Не удалось рассчитать ROC-AUC: {e}")

        # Сохраняем полный датасет в CSV
        output_path = "artifacts/predictions.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Датасет с предсказаниями сохранен в {output_path}")

        # Формируем результат
        result = {
            "message": "Успешно",
            "rows_processed": len(df),
            "has_target": has_target,
            "predictions_file": output_path,
            "timestamp": datetime.now().isoformat()
        }

        if roc_auc is not None:
            result['roc_auc'] = roc_auc

        # Возвращаем полный датасет если запрошен
        if return_full_dataset:
            # Преобразуем DataFrame в список словарей для JSON
            full_dataset = df.to_dict(orient='records')
            result['full_dataset'] = full_dataset
            result['full_dataset_rows'] = len(full_dataset)
            logger.info(f"Возвращен полный датасет с {len(full_dataset)} строками")
        else:
            # Если полный датасет не запрошен, возвращаем только первые 10 строк
            result['sample'] = df.head(10).to_dict(orient='records')

        return result

    except Exception as e:
        logger.error(f"Ошибка обработки CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-predictions", summary="Скачать CSV с предсказаниями", tags=["Prediction"])
async def download_predictions():
    output_path = "artifacts/predictions.csv"
    if os.path.exists(output_path):
        return FileResponse(
            output_path,
            media_type='text/csv',
            filename=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        raise HTTPException(status_code=404, detail="Файл с предсказаниями не найден")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "ML Mortgage Prediction",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": [
            "POST /upload-model",
            "POST /predict",
            "POST /predict-from-csv",
            "GET /download-predictions"
        ]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")

    return {
        "type": type(pipeline).__name__,
        "steps": list(pipeline.named_steps.keys()) if hasattr(pipeline, 'named_steps') else []
    }


if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/ui", tags=["UI"])
async def serve_ui():
    ui_path = "frontend/index.html"
    if os.path.exists(ui_path):
        return FileResponse(ui_path, headers={"Cache-Control": "no-cache"})
    else:
        return {"message": "Frontend не найден"}


if __name__ == "__main__":
    logger.info("Запуск сервера...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)