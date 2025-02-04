# very Small Language Model (vSLM)

## About
A short project to implement a small GPT like model from scratch.
Most of it is inspired by minBPE and nanoGPT.

## How to run the project:
1. Create and activate a virtual environment (optional)
```bash
python -m venv project_venv
source project_venv/bin/activate
```

2. Setup the project and download the requirements
```bash
pip install -e .
pip install -r requirements.txt
```
3. Run the code
```bash
python main.py --training_iterations=5000 --text=shakespeare --train_model=False --task=generation
```

4. In coming:
   * Fine-tuning (LoRA + RLHF)

5. Done:
   * Tokenizer (byte and character level)
   * Full Transformer architecture (Encoder + Decoder)
   * Training and inference pipeline for generation (Lorem Ipsum and Shakespeare)
   * Training and inference pipeline for translation (en → fr)