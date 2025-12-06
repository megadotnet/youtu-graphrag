# ==========================================
# Dockerfile for Youtu-GraphRAG
# ==========================================

# 1. Base Image
# Use an official Python 3.10 slim image to keep the final image size optimized.
FROM python:3.10

# 2. Environment Variables
# - PYTHONUNBUFFERED=1: Ensures Python output is sent straight to terminal (e.g. your container logs) without being buffered.
# - PIP_NO_CACHE_DIR=1: Disables the pip cache to reduce image size.
# - PIP_DISABLE_PIP_VERSION_CHECK=1: Suppresses the "new version of pip" warning.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 3. System Dependencies
# Install necessary system packages:
# - git: For cloning repositories or installing git-based pip packages.
# - default-jre: Java Runtime Environment, required for Apache Tika (document parsing).
# - antiword: Utility to convert MS Word (.doc) files to text.
# - libreoffice-writer/calc: Used for high-fidelity document conversion (doc/docx to text).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        default-jre \
        antiword \
        libreoffice-writer \
        libreoffice-calc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Working Directory
# Set the working directory inside the container.
WORKDIR /youtu_graphrag

# 5. Copy Files
# Copy the current directory contents into the container at /youtu_graphrag.
COPY . /youtu_graphrag/

# 6. Permissions
# Ensure the startup script is executable.
RUN chmod +x start.sh

# 7. Python Dependencies & Models
# - Install Python dependencies from requirements.txt.
# - Download the spaCy English model (en_core_web_lg) by default.
#   Note: If your application is primarily for Chinese, you might want to switch this to 'zh_core_web_lg'.
RUN pip install -r requirements.txt && python -m spacy download en_core_web_lg

# 8. Ports
# Expose port 8000 for the backend server (FastAPI/Uvicorn).
EXPOSE 8000

# 9. Entrypoint
# Define the default command to run when the container starts.
CMD ["sh", "-c", "./start.sh"]
