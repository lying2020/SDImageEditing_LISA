#!/bin/bash
# Conda环境设置脚本

ENV_NAME="sd_impainting"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.7"  # 或 "12.1"

echo "=========================================="
echo "创建Conda环境: $ENV_NAME"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装"
    echo "请先安装Miniconda或Anaconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 创建环境
echo "1. 创建conda环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo "2. 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 安装CUDA工具包
echo "3. 安装CUDA工具包..."
if [ "$CUDA_VERSION" == "11.7" ]; then
    conda install -c conda-forge cudatoolkit=11.7 -y
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
elif [ "$CUDA_VERSION" == "12.1" ]; then
    conda install -c nvidia cuda-toolkit=12.0 -y
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# 安装bitsandbytes
echo "4. 安装bitsandbytes..."
conda install -c conda-forge bitsandbytes -y || pip install bitsandbytes

# 安装其他依赖
echo "5. 安装其他依赖..."
# pip install transformers accelerate opencv-python numpy

echo ""
echo "=========================================="
echo "✅ 环境创建完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo "  python3 chat.py --load_in_4bit=False --precision=fp32"
echo ""
