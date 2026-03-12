FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run model download script
COPY download_model.py .
RUN python download_model.py

# Copy handler
COPY handler.py .
CMD ["python", "-u", "handler.py"]
