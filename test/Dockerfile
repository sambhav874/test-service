# Use Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /test

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Detect GPU and install CUDA if available
RUN if command -v nvidia-smi >/dev/null 2>&1; then \
    apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Upgrade pip in the virtual environment
RUN $VIRTUAL_ENV/bin/pip install --upgrade pip

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install FAISS and CUDA-dependent packages based on GPU availability
RUN if command -v nvidia-smi >/dev/null 2>&1; then \
    $VIRTUAL_ENV/bin/pip install faiss-gpu && \
    # Add any other GPU-specific packages here \
    echo "GPU environment detected - installing GPU dependencies"; \
    else \
    $VIRTUAL_ENV/bin/pip install faiss-cpu && \
    # Add any CPU-specific packages here \
    echo "CPU environment detected - installing CPU dependencies"; \
    fi

# Install huggingface_hub explicitly
RUN pip install huggingface_hub

# Install Python dependencies
RUN $VIRTUAL_ENV/bin/pip install -r requirements.txt

# Create directories for contracts and cleaned outputs
RUN mkdir -p ./contracts ./cleaned_contracts ./final_extracted_data

# Copy the actual .env file if it exists, otherwise use template
COPY .env* ./

# Copy the rest of the application
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]