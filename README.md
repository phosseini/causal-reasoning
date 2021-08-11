# Commonsense Causal Reasoning


## Converting ATOMIC-to-Text
Triples in ATOMIC are stored in form of: `(subject, relation, target)`. We convert (verbalize) these triples to natural language text to later use them in training/fine-tuning some Pretrained Language Models (PLMs).
#### Steps:
1. Download ATOMIC 2020 [here](https://allenai.org/data/atomic-2020), put it in the `/data` folder, and unzip it.
2. Run the the following code: [`src/convert_atomic_to_text.py`](https://github.com/phosseini/causal-reasoning/blob/main/src/convert_atomic_to_text.py)
3. Output will be stored as `.txt` and `.csv` files in the `/data` folder.
