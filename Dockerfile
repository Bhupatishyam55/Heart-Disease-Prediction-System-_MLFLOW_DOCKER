# ------------ 1. Base Image ------------
FROM python:3.10-slim

# ------------ 2. Set Working Directory ------------
WORKDIR /app

# ------------ 3. Install System Dependencies ------------
# (needed for numpy, pandas, scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------ 4. Copy Requirements and Install ------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ------------ 5. Copy Project Files ------------
COPY . .

# ------------ 6. Expose Port ------------
EXPOSE 8000

# ------------ 7. Run FastAPI Server ------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
