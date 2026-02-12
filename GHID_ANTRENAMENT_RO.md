# Ghid Antrenament Dataset Toxicitate - Politicieni Români

## ⚠️ IMPORTANT: Despre llama.cpp

**llama.cpp NU este potrivit pentru antrenament!**

- llama.cpp este optimizat pentru **INFERENȚĂ** (rulare modele cuantizate)
- NU suportă fine-tuning sau antrenament
- Este util doar pentru rularea rapidă a modelelor antrenate, nu pentru antrenare

**Pentru antrenament ai nevoie de:**
- PyTorch + Transformers (deja instalat în proiect)
- CUDA/GPU pentru antrenament eficient
- TRL (Training Reinforcement Learning) - deja în dependențe

---

## 🎯 Modele Recomandate pentru Antrenament în Limba Română

### 1. **DeepSeek-R1-Distill-Qwen-7B** (CURRENT - BUN)
- ✅ Deja configurat în proiect
- ✅ Suport bun pentru română
- ✅ Model de raționament (reasoning)
- ✅ 7B parametri - eficient pentru GPU

### 2. **Qwen2.5-7B-Instruct** (RECOMANDAT)
- ✅ Excelent suport multilingv (inclusiv română)
- ✅ Model instruct optimizat
- ✅ Performanță bună pe task-uri de clasificare
- ✅ Disponibil pe HuggingFace: `Qwen/Qwen2.5-7B-Instruct`

### 3. **OpenLLM-Ro** (SPECIFIC ROMÂNĂ)
- ✅ Dezvoltat de Politehnica București
- ✅ Antrenat pe milioane de documente românești
- ⚠️ Trebuie verificat disponibilitatea pe HuggingFace
- 🔗 Căutare: `ai-romania` sau `OpenLLM-Ro` pe HuggingFace

### 4. **Llama 3.1 8B Instruct** (ALTERNATIVĂ)
- ✅ Suport multilingv bun
- ✅ Model instruct robust
- ✅ Disponibil: `meta-llama/Llama-3.1-8B-Instruct`

---

## 📊 Structura Dataset-ului Actual

Din fișierul `judge_reasoning.jsonl` văd că ai:
- Texte în limba română despre politicieni
- Evaluări de la judge (rating 1-10)
- Completări de la model pentru clasificare toxic/non-toxic

**Format actual:**
```json
{
  "call": 1,
  "rating": 5,
  "judge_response": "",
  "original_text": "text în română...",
  "model_completion": "răspuns model..."
}
```

---

## 🚀 Pași pentru Augmentare Dataset

### Pasul 1: Verifică Dataset-ul Augmentat Există

```bash
cd /home/miriam/torch_rar_project
uv run python3 -c "import pandas as pd; df = pd.read_parquet('output/augmented_dataset.parquet'); print(f'Dataset: {len(df)} samples'); print(f'Columns: {list(df.columns)}')"
```

### Pasul 2: Rulează Augmentare (dacă nu există sau vrei mai mult)

```bash
# Augmentare cu 100 de sample-uri
uv run python main.py run --limit 100

# Sau cu rubrics predefinite (mai rapid)
uv run python main.py run --limit 100 --predefined-rubrics

# Sau doar implicit reward (mai rapid)
uv run python main.py run --limit 100 --reward-method implicit
```

### Pasul 3: Verifică Output-ul

Dataset-ul augmentat va fi salvat în:
- `output/augmented_dataset.parquet` (format recomandat)
- Sau `output/augmented_dataset.json`

---

## 🎓 Antrenament Model

### Opțiunea 1: Antrenament cu Modelul Actual (DeepSeek-R1)

```bash
# Antrenament de bază (2 epochs, hybrid reward)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py

# Antrenament cu mai multe epochs
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --epochs 3 \
    --lr 1e-5 \
    --batch-size 2

# Antrenament cu rule-based reward (mai rapid, fără judge)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --reward-mode rule_based \
    --epochs 2
```

### Opțiunea 2: Antrenament cu Qwen2.5 (RECOMANDAT pentru română)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --lr 5e-6 \
    --reward-mode hybrid
```

### Opțiunea 3: Antrenament cu Llama 3.1

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --epochs 3 \
    --lr 5e-6
```

### Configurare GPU

Proiectul este configurat pentru 2 GPU-uri:
- **GPU 0**: Antrenament (model + LoRA + optimizer)
- **GPU 1**: Judge (DeepSeek-R1:70b via Ollama)

Dacă ai doar 1 GPU:
```bash
# Folosește doar rule-based reward (nu necesită judge)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --reward-mode rule_based \
    --epochs 2
```

---

## 📈 Evaluare Model

### Evaluare Model Antrenat

```bash
# Evaluează modelul final antrenat
uv run python scripts/evaluate.py --model ./checkpoints/final

# Evaluează un checkpoint specific
uv run python scripts/evaluate.py --model ./checkpoints/checkpoint-250

# Evaluează cu mai puține sample-uri (test rapid)
uv run python scripts/evaluate.py --model ./checkpoints/final --max-samples 20
```

### Comparare Baseline vs Antrenat

```bash
# Compară modelul antrenat cu baseline-ul
uv run python scripts/evaluate.py \
    --model ./checkpoints/final \
    --compare-baseline deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### Evaluare Baseline (model neantrenat)

```bash
uv run python scripts/evaluate.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --baseline
```

---

## 🔧 Configurare pentru Limba Română

### Verifică Configurația Actuală

Fișierul `config/settings.yaml` este deja configurat pentru:
- Dataset românesc: `olimpia20/toxicity-dataset-ro-master`
- Prompt-uri pentru context românesc
- Rubrics specifice pentru discurs politic românesc

### Prompt-uri Personalizate

Prompt-urile sunt în `prompts/toxicity/`:
- `rubric_system.jinja2` - System prompt pentru generare rubrics
- `rubric_user.jinja2` - User prompt pentru rubrics
- `implicit_eval_system.jinja2` - System prompt pentru evaluare

Toate sunt deja optimizate pentru:
- ✅ Limba română
- ✅ Context politic românesc
- ✅ Politicieni români (Iohannis, Ciolacu, PSD, AUR, etc.)

---

## 📋 Checklist Antrenament

- [ ] Verifică că dataset-ul augmentat există (`output/augmented_dataset.parquet`)
- [ ] Verifică GPU disponibil: `nvidia-smi`
- [ ] Alege modelul pentru antrenament (recomandat: Qwen2.5 sau DeepSeek-R1)
- [ ] Rulează antrenament: `scripts/train.py`
- [ ] Monitorizează progresul (checkpoints în `checkpoints/`)
- [ ] Evaluează modelul: `scripts/evaluate.py`
- [ ] Compară cu baseline pentru a vedea îmbunătățiri

---

## 🎯 Recomandări Finale

### Pentru Cel Mai Bun Rezultat în Română:

1. **Model**: `Qwen/Qwen2.5-7B-Instruct` sau `DeepSeek-R1-Distill-Qwen-7B`
2. **Dataset**: Augmentează cel puțin 500-1000 de sample-uri
3. **Antrenament**: 
   - 3 epochs
   - Learning rate: 5e-6
   - Hybrid reward (rule-based + judge)
   - LoRA (eficient, păstrează modelul original)

### Pentru Antrenament Rapid (Test):

1. **Model**: `DeepSeek-R1-Distill-Qwen-7B` (deja configurat)
2. **Dataset**: 100-200 sample-uri augmentate
3. **Antrenament**:
   - 2 epochs
   - Rule-based reward (mai rapid)
   - LoRA

---

## ❓ FAQ

**Q: Pot folosi llama.cpp pentru antrenament?**
A: NU. llama.cpp este doar pentru inferență. Pentru antrenament folosește PyTorch/Transformers.

**Q: Care model e cel mai bun pentru română?**
A: Qwen2.5-7B-Instruct sau OpenLLM-Ro (dacă e disponibil).

**Q: Cât timp durează antrenamentul?**
A: Depinde de GPU și numărul de sample-uri. Pe H200: ~2-4 ore pentru 1000 sample-uri, 3 epochs.

**Q: Pot antrena fără GPU?**
A: Teoretic da, dar va fi foarte lent. Recomand GPU cu cel puțin 16GB VRAM.

---

## 📚 Resurse

- Dataset: `olimpia20/toxicity-dataset-ro-master` pe HuggingFace
- Model actual: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Documentație: `docs/README_TECHNICAL.md`

