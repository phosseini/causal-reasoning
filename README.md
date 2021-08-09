# Commonsense Causal Reasoning


## Converting ATOMIC-to-Text
Triples in ATOMIC are stored in form of: `(subject, relation, target)`. We convert these triples to natural language text to later use them in training/fine-tuning some Pretrained Language Models (PLMs). 
* Before converting triples, download ATOMIC 2020 [here](https://allenai.org/data/atomic-2020), put it in the `/data` folder, and unzip it.
* Now, run the the following code: `src/convert_atomic_to_text.py`
* Output will be a `.txt` and `.csv` file.
