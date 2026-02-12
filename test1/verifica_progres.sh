#!/bin/bash
# Script pentru verificare progres antrenament

cd /home/miriam/torch_rar_project/test1

echo "=========================================="
echo "  Verificare Progres Antrenament"
echo "=========================================="
echo ""

# Verifică dacă procesul rulează
if pgrep -f "train_test1.py" > /dev/null; then
    echo "✅ Antrenament în curs..."
    echo ""
    
    # Verifică GPU-uri
    echo "Utilizare GPU:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    
    # Verifică checkpoint-uri
    if [ -d "checkpoints" ]; then
        CHECKPOINTS=$(ls -d checkpoints/checkpoint-* 2>/dev/null | wc -l)
        if [ "$CHECKPOINTS" -gt 0 ]; then
            echo "✅ Checkpoints găsite: $CHECKPOINTS"
            echo "Ultimul checkpoint:"
            ls -d checkpoints/checkpoint-* 2>/dev/null | tail -1 | xargs basename
        else
            echo "⏳ Încă nu există checkpoint-uri (antrenamentul tocmai a început)"
        fi
    fi
    echo ""
    
    # Ultimele linii din log
    if [ -f "training_log.txt" ]; then
        echo "Ultimele linii din log:"
        tail -5 training_log.txt | sed 's/^/  /'
    fi
else
    echo "❌ Antrenamentul NU rulează"
    
    # Verifică dacă s-a terminat cu succes
    if [ -d "checkpoints/final" ]; then
        echo ""
        echo "✅ Antrenament terminat cu succes!"
        echo "Model final: checkpoints/final/"
        
        if [ -f "checkpoints/eval_results.json" ]; then
            echo ""
            echo "Rezultate evaluare:"
            cat checkpoints/eval_results.json | python3 -m json.tool 2>/dev/null || cat checkpoints/eval_results.json
        fi
    fi
fi

echo ""
echo "Pentru log complet: tail -f training_log.txt"

