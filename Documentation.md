# Time Series Forecasting System API

**Version:** 0.1.0

**Framework:** FastAPI

This project provides an end-to-end pipeline for time series analysis, allowing administrators to manage datasets and evaluate models, while providing end-users with predictive insights based on regional data.

## System Architecture

The API is divided into two primary sections: **Admin APIs** for pipeline management and **User APIs** for consuming the forecasting models.

---

## Admin Endpoints

These endpoints are designed for the data science workflow, including data ingestion and model selection.

### 1. Upload Dataset

* **Endpoint:** `POST /upload-data`
* **Description:** Uploads a raw CSV or data file to the system for processing.
* `file`: The binary file containing the time series data.


* **Response:** `200 OK` on successful upload.

### 2. Evaluate Models

* **Endpoint:** `GET /evaluate-models`
* **Description:** Triggers the evaluation suite (comparing ARIMA, Prophet, and XGBoost models). It calculates performance metrics like RMSE , MAE and MAPE to determine the most accurate model for the current dataset.

### 3. Save Best Model

* **Endpoint:** `GET /save-best-model`
* **Description:** Finalizes the pipeline by persisting the highest-performing model from the evaluation stage for production use.

---

## User Endpoints

These endpoints are exposed for client-side applications to retrieve predictions.

### 1. Generate Forecast

* **Endpoint:** `POST /forecast`
* **Description:** Returns time series predictions for a specific US state over a defined period.
* **Request Schema (`PredictionInput`):**

| Field | Type | Description | Default | Constraints |
| --- | --- | --- | --- | --- |
| `state` | string | The US State to forecast for. | `New York` | Must be a valid US state name. |
| `weeks` | integer | Number of weeks to predict into the future. | N/A | Min: 1, Max: 80 |

**Example Request:**

```json
{
  "state": "New York",
  "weeks": 8
}

```

---

## 🧪 How to Run

1. **Install dependencies:** `uv sync`
2. **Start the server:** `uv run fastapi dev main.py`
3. **Access Interactive Docs:** Navigate to `/docs` to view the Swagger UI or `/redoc` for ReDoc documentation.
