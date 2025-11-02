# API Documentation

## Retina-Based Heart Disease Predictor API

Base URL: `http://localhost:5000`

---

## Endpoints

### 1. Web Interface
**GET** `/`

Returns the main web interface HTML page.

**Response**: HTML page with upload form

---

### 2. Prediction Endpoint
**POST** `/predict`

Upload a retinal fundus image to get heart disease risk prediction.

#### Request

**Content-Type**: `multipart/form-data`

**Parameters**:
- `image` (file, required): Retinal fundus image (PNG, JPG, JPEG)
  - Max size: 16MB
  - Supported formats: `.png`, `.jpg`, `.jpeg`

#### Response

**Success** (200 OK):
```json
{
  "success": true,
  "heart_disease_risk": 65.23,
  "has_disease": true,
  "prediction": 0.6523,
  "message": "High risk of heart disease detected. Risk level: 65.23%"
}
```

**Error** (400 Bad Request):
```json
{
  "error": "No image file provided"
}
```

**Error** (500 Internal Server Error):
```json
{
  "error": "Model not loaded. Please train the model first."
}
```

#### Example (cURL)
```bash
curl -X POST \
  -F "image=@retinal_image.jpg" \
  http://localhost:5000/predict
```

#### Example (Python)
```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("retinal_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

if result["success"]:
    print(f"Risk: {result['heart_disease_risk']}%")
    print(f"Has Disease: {result['has_disease']}")
else:
    print(f"Error: {result['error']}")
```

#### Example (JavaScript)
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log(`Risk: ${data.heart_disease_risk}%`);
    } else {
        console.error(`Error: ${data.error}`);
    }
});
```

---

### 3. Health Check
**GET** `/health`

Check server and model status.

#### Response

**Success** (200 OK):
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model_path": "models/retina_heart_disease_model.h5"
}
```

**Example**:
```bash
curl http://localhost:5000/health
```

---

## Response Fields

### Prediction Response

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was successful |
| `heart_disease_risk` | float | Risk percentage (0-100) |
| `has_disease` | boolean | Whether disease is detected (risk > 50%) |
| `prediction` | float | Raw prediction value (0-1) |
| `message` | string | Human-readable result message |

### Health Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Server status ("healthy") |
| `model_status` | string | Model loading status ("loaded" or "not loaded") |
| `model_path` | string | Path to the model file |

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 500 | Internal Server Error - Server/model error |

---

## Rate Limiting

Currently: No rate limiting (consider adding for production)

---

## Best Practices

1. **Image Requirements**:
   - Use retinal fundus images
   - Recommended size: 224x224 pixels or larger
   - Supported formats: PNG, JPG, JPEG

2. **Error Handling**:
   - Always check `success` field in response
   - Handle network errors
   - Validate file before upload

3. **Performance**:
   - Compress images before upload if possible
   - Cache predictions when appropriate
   - Use health endpoint to check server status

---

## Security Notes

⚠️ **For Production**:
- Add authentication/API keys
- Implement rate limiting
- Use HTTPS/SSL
- Validate and sanitize all inputs
- Set up firewall rules

---

## Testing

### Test with sample image
```bash
# Download a test image first, then:
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

### Health check
```bash
curl http://localhost:5000/health
```

---

**API Version**: 1.0
**Last Updated**: November 2025

