# very Small Language Model (vSLM)

## About
A short project to implement a (very) small language model from scratch, from the tokenizer to the fine-tuning.

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
python main.py --training_iterations=5000 --text=shakespeare --train_model=False 
```

4. In coming
   * Fine-tuning
   * Encoding-Decoding architecture 
