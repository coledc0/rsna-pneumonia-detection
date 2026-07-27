from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.predict import load_model, predict
from api.schemas import PredictionResponse, HealthResponse

import boto3
from botocore.exceptions import ClientError

from src.model_utils import find_best_run

import os

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# global state - loaded once at startup, reused across requests
config = load_config()
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    local_weights_path = Path(os.environ.get('LOCAL_WEIGHTS_PATH', 'weights/best.pt'))
    local_weights_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_name = os.environ.get('S3_BUCKET_NAME')

    if bucket_name:
        # production path: download from S3
        if not local_weights_path.exists():
            s3_key = os.environ.get('S3_WEIGHTS_KEY', 'best.pt')
            print(f'Downloading weights from s3://{bucket_name}/{s3_key}')
            try:
                s3 = boto3.client('s3')
                s3.download_file(bucket_name, s3_key, str(local_weights_path))
                print('Weights downloaded successfully')
            except ClientError as e:
                raise RuntimeError(f'Failed to download weights from S3: {e}')
        else:
            print(f'Using cached weights at {local_weights_path}')
    else:
        # local dev fallback: scan runs/ folder for best local checkpoint
        print('No S3_BUCKET_NAME set, falling back to local run scanning')
        local_weights_path = find_best_run(Path('runs/detect/weights'))

    print(f'Loading model from {local_weights_path}')
    model_state['model'] = load_model(local_weights_path)
    model_state['model_path'] = str(local_weights_path)
    print('Model loaded successfully')

    yield
    model_state.clear()

app = FastAPI(
    title = 'RSNA Pneumonia Detection API',
    description = 'Detects pneumonia in chest X-rays using a YOLOv8 model trained on the RSNA dataset.',
    version = '1.0.0',
    lifespan = lifespan,
)

# allow requests from any origin (would be restricted in a real production deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/', response_model=HealthResponse)
async def root():
    '''
    Health check endpoint.
    '''
    return HealthResponse(
        status='ok',
        model_loaded='model' in model_state,
        model_path=model_state.get('model_path', 'not loaded'),
    )

@app.get('/health', response_model = HealthResponse)
async def health():
    '''
    Explicit health check endpoint, standard for production APIs.
    '''
    return HealthResponse(
        status='ok',
        model_loaded = 'model' in model_state,
        model_path = model_state.get('model_path', 'not loaded'),
    )

@app.post('/predict', response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    '''
    Upload a chest X-ray image (PNG/JPG) and receive a pneumonia detection prediction with bounding boxes.
    '''
    if 'model' not in model_state:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    # validates file type
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {file.content_type}.' f'Upload PNG or JPEG.'
        )
    try:
        image_bytes = await file.read()

        result = predict(
            model=          model_state['model'],
            image_input=    image_bytes,
            img_size=       config['data']['image_size'],
            conf_threshold= config['api']['confidence_threshold'],
        )

        return PredictionResponse(**result, filename=file.filename)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'api.main:app',
        host=config['api']['host'],
        port=config['api']['port'],
        reload=True
    )