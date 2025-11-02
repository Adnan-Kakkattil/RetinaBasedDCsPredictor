# Production Deployment Guide

## ✅ Production Server Started

The Retina-Based Heart Disease Predictor is now running in production mode.

## Server Information

- **URL**: http://localhost:5000
- **Network Access**: http://0.0.0.0:5000 (accessible from network)
- **Mode**: Production (debug=False, threaded=True)
- **Model**: Loaded and ready for predictions

## Features

### ✅ Production-Ready Features

1. **Model Loaded**: 97 MB trained model ready for predictions
2. **Threaded Server**: Handles multiple requests simultaneously
3. **Error Handling**: Comprehensive error handling for edge cases
4. **File Upload**: Secure image upload with validation
5. **API Endpoints**: RESTful API for predictions

## API Endpoints

### 1. Web Interface
- **GET** `/` - Main web interface
  - Upload retinal fundus images
  - Get visual predictions with risk percentage

### 2. Prediction API
- **POST** `/predict`
  - **Input**: Multipart form data with `image` file
  - **Output**: JSON with prediction results
  ```json
  {
    "success": true,
    "heart_disease_risk": 65.23,
    "has_disease": true,
    "prediction": 0.6523,
    "message": "High risk of heart disease detected. Risk level: 65.23%"
  }
  ```

### 3. Health Check
- **GET** `/health`
  - **Output**: Server and model status
  ```json
  {
    "status": "healthy",
    "model_status": "loaded",
    "model_path": "models/retina_heart_disease_model.h5"
  }
  ```

## Usage

### Web Interface

1. Open browser: http://localhost:5000
2. Drag & drop or click to upload retinal fundus image
3. Click "Predict Heart Disease Risk"
4. View results with risk percentage and recommendations

### API Usage (cURL)

```bash
# Make a prediction
curl -X POST -F "image=@path/to/retinal_image.jpg" http://localhost:5000/predict

# Check health
curl http://localhost:5000/health
```

### API Usage (Python)

```python
import requests

# Upload and predict
with open('retinal_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
result = response.json()
print(f"Risk: {result['heart_disease_risk']}%")
```

## Production Considerations

### 1. Security
- ✅ File type validation
- ✅ File size limits (16MB max)
- ✅ Secure filename handling
- ⚠️ Consider adding:
  - Authentication/API keys
  - Rate limiting
  - HTTPS/SSL certificates
  - Input sanitization

### 2. Performance
- ✅ Threaded server for concurrent requests
- ✅ Model preloaded (faster predictions)
- ✅ Optimized image preprocessing
- ⚠️ For high traffic, consider:
  - Gunicorn/uWSGI with multiple workers
  - Nginx reverse proxy
  - Load balancing
  - Caching

### 3. Monitoring
- Health check endpoint available
- Error logging
- ⚠️ Consider adding:
  - Application monitoring (e.g., Prometheus)
  - Log aggregation
  - Performance metrics
  - Uptime monitoring

### 4. Scalability
- Current setup: Single server
- ⚠️ For scale, consider:
  - Container deployment (Docker)
  - Kubernetes orchestration
  - Cloud deployment (AWS, Azure, GCP)
  - Auto-scaling

## Deployment Options

### Option 1: Current Setup (Development/Testing)
```bash
python app.py --production
```

### Option 2: Gunicorn (Recommended for Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Option 4: Windows Service
- Use NSSM (Non-Sucking Service Manager) to run as Windows service
- Or use Task Scheduler for auto-start

## Stopping the Server

Press `CTRL+C` in the terminal running the server.

## Restarting

```bash
# Using batch file
start_production.bat

# Or directly
python app.py --production
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Model Not Found
```bash
# Train model first
python src/train.py
```

### Memory Issues
- Reduce batch size in config.py
- Use smaller model (MobileNetV2 instead of ResNet50)
- Enable GPU support for better performance

## Network Access

To access from other devices on your network:
1. Find your local IP: `ipconfig` (look for IPv4 Address)
2. Access via: `http://<your-ip>:5000`
3. Ensure firewall allows port 5000

## Next Steps

1. ✅ Server is running
2. ⚠️ Test predictions via web interface
3. ⚠️ Set up monitoring/logging
4. ⚠️ Configure firewall rules
5. ⚠️ Consider SSL/HTTPS for production
6. ⚠️ Set up backup/restore procedures

---

**Status**: ✅ Production server is running and ready for predictions!

