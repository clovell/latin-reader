import urllib.request
import json
import os
import re
from collections import defaultdict

def main():
    url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-Perseus/master/la_perseus-ud-train.conllu"
    print(f"Downloading treebank from {url}...")
    lines = urllib.request.urlopen(url).read().decode('utf-8').splitlines()

    sentences = defaultdict(list)
    
    current_author = None
    current_text = None
    current_tokens = []

    # Known PHI code mappings for the Perseus treebank
    author_map = {
        'phi0448': 'Caesar',
        'phi0474': 'Cicero',
        'phi0690': 'Vergil',
        'phi0959': 'Ovid',
        'phi0975': 'Phaedrus',
        'phi0972': 'Petronius',
        'phi1002': 'Quintilian',
        'phi1017': 'Seneca',
        'phi0472': 'Catullus',
        'phi0620': 'Propertius',
        'phi0631': 'Sallust',
        'phi1048': 'Suetonius',
        'phi0119': 'Plautus',
        'phi1221': 'Augustus',
        'phi1348': 'Suetonius',
        'phi1351': 'Tacitus',
    }

    print("Parsing CoNLL-U format...")
    for line in lines:
        line = line.strip()
        if not line:
            # End of sentence
            if current_author and current_text and current_tokens:
                # Resolve HEAD text
                for t in current_tokens:
                    head_idx = int(t['head'])
                    if head_idx == 0:
                        t['head_text'] = t['text']
                    else:
                        # 1-based indexing in CoNLL-U, find the token
                        head_tok = next((x for x in current_tokens if x['id'] == head_idx), None)
                        t['head_text'] = head_tok['text'] if head_tok else "ROOT"
                        
                sentences[current_author].append({
                    "text": current_text,
                    "tokens": current_tokens
                })
            current_author = None
            current_text = None
            current_tokens = []
            continue

        if line.startswith('# sent_id ='):
            sent_id = line.split('=')[1].strip()
            # Extract phiXXXX
            match = re.search(r'(phi\d{4})', sent_id)
            if match:
                author_code = match.group(1)
                current_author = author_map.get(author_code, author_code)
            else:
                current_author = 'Unknown'
        elif line.startswith('# text ='):
            current_text = line.split('=')[1].strip()
            current_text = re.sub(r'\s+', ' ', current_text)
        elif not line.startswith('#'):
            parts = line.split('\t')
            # CoNLL-U columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
            if len(parts) >= 8:
                try:
                    tok_id = int(parts[0])  # ignore multi-word tokens for simplification if they have hyphens
                    current_tokens.append({
                        "id": tok_id,
                        "text": parts[1],
                        "lemma": parts[2],
                        "pos": parts[3],
                        "tag": parts[4],
                        "morph": parts[5],
                        "head": parts[6],
                        "dep": parts[7]
                    })
                except ValueError:
                    pass # multiword token range e.g. "1-2"

    # Prepare output path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'static', 'data')
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'perseus_sentences.json')

    print(f"Writing to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    total_sentences = sum(len(v) for v in sentences.values())
    print(f"Saved {total_sentences} sentences across {len(sentences)} authors.")

if __name__ == "__main__":
    main()
