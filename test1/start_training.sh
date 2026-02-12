#!/bin/bash
# Script pentru pornire antrenament cu ambele GPU-uri

cd /home/miriam/torch_rar_project/test1

echo "=========================================="
echo "  Pornire Antrenament - 2 GPU-uri"
echo "=========================================="
echo ""

# Verifică GPU-urile
echo "GPU-uri disponibile:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Setează ambele GPU-uri vizibile
export CUDA_VISIBLE_DEVICES=0,1

echo "✓ CUDA_VISIBLE_DEVICES=0,1 (ambele GPU-uri)"
echo "✓ Antrenamentul va folosi automat ambele GPU-uri"
echo ""

# Pornește antrenamentul
echo "Pornire antrenament..."
echo ""

CUDA_VISIBLE_DEVICES=0,1 uv run python train_test1.py 2>&1 | tee training_log.txt

echo ""
echo "Antrenament terminat!"
echo "Verifică training_log.txt pentru detalii"

