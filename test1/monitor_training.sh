#!/bin/bash
# Script pentru monitorizare antrenament

echo "Monitorizare antrenament test1..."
echo ""

while true; do
    # Verifică dacă procesul rulează
    if pgrep -f "train_test1.py" > /dev/null; then
        echo "[$(date '+%H:%M:%S')] Antrenament în curs..."
        
        # Verifică dacă există checkpoint-uri
        if [ -d "checkpoints" ]; then
            CHECKPOINTS=$(ls -d checkpoints/checkpoint-* 2>/dev/null | wc -l)
            if [ "$CHECKPOINTS" -gt 0 ]; then
                echo "  Checkpoints găsite: $CHECKPOINTS"
                ls -d checkpoints/checkpoint-* 2>/dev/null | tail -1 | xargs basename
            fi
        fi
        
        # Verifică GPU usage
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "  GPU: %s%% utilizare, %s/%s MB VRAM\n", $1, $2, $3}'
        fi
    else
        echo "[$(date '+%H:%M:%S')] Antrenament terminat!"
        
        # Verifică rezultatele finale
        if [ -f "checkpoints/eval_results.json" ]; then
            echo ""
            echo "=== REZULTATE FINALE ==="
            cat checkpoints/eval_results.json | python3 -m json.tool 2>/dev/null || cat checkpoints/eval_results.json
        fi
        
        if [ -d "checkpoints/final" ]; then
            echo ""
            echo "✓ Model final salvat în: checkpoints/final"
        fi
        
        break
    fi
    
    sleep 30
done

