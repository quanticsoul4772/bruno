# Heretic - GPU Cloud Docker Image
# Works on RunPod, Vast.ai, and local GPU setups
#
# Build: docker build -t quanticsoul4772/heretic .
# Run:   docker run --gpus all -it quanticsoul4772/heretic heretic <model>
#
# For RunPod/Vast.ai, use this image directly or push to Docker Hub

# =============================================================================
# Base Image: PyTorch with CUDA 12.4 (matches RunPod default)
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Labels for container registries
LABEL org.opencontainers.image.title="Heretic LLM Abliteration"
LABEL org.opencontainers.image.description="Automatic censorship removal for language models"
LABEL org.opencontainers.image.source="https://github.com/quanticsoul4772/heretic"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

# =============================================================================
# Environment Configuration
# =============================================================================

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# HuggingFace cache configuration (persistent on /workspace)
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA settings for better performance
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Python Dependencies & Heretic Installation
# =============================================================================

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install heretic from the fork with all improvements
# (--auto-select, --hf-upload, Optuna pruning, etc.)
RUN pip install --no-cache-dir git+https://github.com/quanticsoul4772/heretic.git

# Install additional useful tools
RUN pip install --no-cache-dir \
    huggingface-hub[cli] \
    hf-transfer

# =============================================================================
# Workspace Setup
# =============================================================================

# Create workspace directory (standard for RunPod/Vast.ai)
RUN mkdir -p /workspace/.cache/huggingface /workspace/models /workspace/outputs

# Set working directory
WORKDIR /workspace

# =============================================================================
# Entrypoint Configuration
# =============================================================================

# Copy entrypoint script and fix Windows line endings
COPY docker-entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (show help)
CMD ["heretic", "--help"]
