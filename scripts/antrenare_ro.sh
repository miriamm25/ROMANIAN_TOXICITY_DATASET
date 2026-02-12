#!/bin/bash
# Script pentru antrenament dataset toxicitate română

set -e

PROJECT_DIR="/home/miriam/torch_rar_project"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  ANTrenament Dataset Toxicitate RO"
echo "=========================================="
echo ""

# Culori pentru output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funcție pentru verificare GPU
check_gpu() {
    echo -e "${YELLOW}Verificare GPU...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""
    else
        echo -e "${RED}ATENȚIE: nvidia-smi nu este disponibil. Antrenamentul va fi foarte lent pe CPU!${NC}"
        echo ""
    fi
}

# Funcție pentru verificare dataset
check_dataset() {
    echo -e "${YELLOW}Verificare dataset augmentat...${NC}"
    if [ -f "output/augmented_dataset.parquet" ]; then
        echo -e "${GREEN}✓ Dataset găsit: output/augmented_dataset.parquet${NC}"
        # Încearcă să citească dimensiunea (necesită pandas)
        if command -v uv &> /dev/null; then
            SIZE=$(uv run python3 -c "import pandas as pd; df = pd.read_parquet('output/augmented_dataset.parquet'); print(len(df))" 2>/dev/null || echo "?")
            echo "  Dimensiune: ~$SIZE sample-uri"
        fi
    else
        echo -e "${RED}✗ Dataset augmentat NU există!${NC}"
        echo "  Rulează mai întâi: uv run python main.py run --limit 100"
        return 1
    fi
    echo ""
}

# Funcție pentru afișare meniu
show_menu() {
    echo "Selectează acțiunea:"
    echo "  1) Verificare setup (GPU, dataset, dependențe)"
    echo "  2) Augmentare dataset (100 sample-uri)"
    echo "  3) Augmentare dataset (500 sample-uri)"
    echo "  4) Antrenament - DeepSeek-R1 (model actual)"
    echo "  5) Antrenament - Qwen2.5-7B (recomandat pentru RO)"
    echo "  6) Antrenament - Llama 3.1 8B"
    echo "  7) Antrenament - Rule-based only (rapid, fără judge)"
    echo "  8) Evaluare model antrenat"
    echo "  9) Comparare baseline vs antrenat"
    echo "  0) Ieșire"
    echo ""
    read -p "Alegere [0-9]: " choice
}

# Funcție pentru antrenament
train_model() {
    local MODEL=$1
    local REWARD_MODE=${2:-hybrid}
    local EPOCHS=${3:-3}
    local LR=${4:-5e-6}
    
    echo -e "${GREEN}Start antrenament cu model: $MODEL${NC}"
    echo "  Reward mode: $REWARD_MODE"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LR"
    echo ""
    
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
        --base-model "$MODEL" \
        --reward-mode "$REWARD_MODE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size 2
}

# Funcție pentru evaluare
evaluate_model() {
    local MODEL_PATH=$1
    echo -e "${GREEN}Evaluare model: $MODEL_PATH${NC}"
    uv run python scripts/evaluate.py --model "$MODEL_PATH"
}

# Main
check_gpu

# Dacă sunt argumente, rulează direct
if [ $# -gt 0 ]; then
    case $1 in
        check)
            check_dataset
            exit 0
            ;;
        augment)
            LIMIT=${2:-100}
            echo -e "${GREEN}Augmentare dataset cu $LIMIT sample-uri...${NC}"
            uv run python main.py run --limit "$LIMIT" --predefined-rubrics
            exit 0
            ;;
        train)
            MODEL=${2:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
            train_model "$MODEL"
            exit 0
            ;;
        eval)
            MODEL_PATH=${2:-"./checkpoints/final"}
            evaluate_model "$MODEL_PATH"
            exit 0
            ;;
        *)
            echo "Utilizare: $0 [check|augment [limit]|train [model]|eval [path]]"
            exit 1
            ;;
    esac
fi

# Meniu interactiv
while true; do
    show_menu
    
    case $choice in
        1)
            check_gpu
            check_dataset
            echo -e "${GREEN}Verificare dependențe...${NC}"
            if command -v uv &> /dev/null; then
                echo "✓ uv instalat"
            else
                echo -e "${RED}✗ uv NU este instalat${NC}"
            fi
            echo ""
            ;;
        2)
            echo -e "${GREEN}Augmentare dataset (100 sample-uri)...${NC}"
            uv run python main.py run --limit 100 --predefined-rubrics
            echo ""
            ;;
        3)
            echo -e "${GREEN}Augmentare dataset (500 sample-uri)...${NC}"
            uv run python main.py run --limit 500 --predefined-rubrics
            echo ""
            ;;
        4)
            if check_dataset; then
                train_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "hybrid" 3 5e-6
            fi
            ;;
        5)
            if check_dataset; then
                train_model "Qwen/Qwen2.5-7B-Instruct" "hybrid" 3 5e-6
            fi
            ;;
        6)
            if check_dataset; then
                train_model "meta-llama/Llama-3.1-8B-Instruct" "hybrid" 3 5e-6
            fi
            ;;
        7)
            if check_dataset; then
                train_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "rule_based" 2 5e-6
            fi
            ;;
        8)
            MODEL_PATH="./checkpoints/final"
            if [ -d "$MODEL_PATH" ]; then
                evaluate_model "$MODEL_PATH"
            else
                echo -e "${RED}Modelul $MODEL_PATH nu există!${NC}"
                echo "Selectează un checkpoint:"
                ls -d checkpoints/checkpoint-* 2>/dev/null | head -5
            fi
            echo ""
            ;;
        9)
            if [ -d "./checkpoints/final" ]; then
                echo -e "${GREEN}Comparare baseline vs antrenat...${NC}"
                uv run python scripts/evaluate.py \
                    --model ./checkpoints/final \
                    --compare-baseline deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
            else
                echo -e "${RED}Modelul antrenat nu există! Antrenează mai întâi.${NC}"
            fi
            echo ""
            ;;
        0)
            echo "Ieșire..."
            exit 0
            ;;
        *)
            echo -e "${RED}Alegere invalidă!${NC}"
            echo ""
            ;;
    esac
done

