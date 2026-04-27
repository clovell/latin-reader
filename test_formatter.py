def reconstruct_entry(stems, pos, morph):
    stems = [s for s in stems if s and s != 'zzz' and s != 'empty']
    if not stems: return ""
    
    parts = morph.split()
    decl_conj = parts[0] if len(parts) > 0 else '0'
    var = parts[1] if len(parts) > 1 else '0'
    
    res = []
    
    if pos == 'N':
        gender = parts[2] if len(parts) > 2 else ''
        gen_str = f", {gender.lower()}." if gender and gender != 'X' else ""
        if decl_conj == '1':
            res.append(stems[0] + "a")
            if len(stems) > 1: res.append(stems[1] + "ae" + gen_str)
        elif decl_conj == '2':
            if var == '1': res.append(stems[0] + "us")
            elif var == '2': res.append(stems[0] + "um")
            elif var == '3': res.append(stems[0]) # puer
            else: res.append(stems[0] + "us")
            if len(stems) > 1: res.append(stems[1] + "i" + gen_str)
        elif decl_conj == '3':
            res.append(stems[0])
            if len(stems) > 1: res.append(stems[1] + "is" + gen_str)
        elif decl_conj == '4':
            if var == '2': res.append(stems[0] + "u")
            else: res.append(stems[0] + "us")
            if len(stems) > 1: res.append(stems[1] + "us" + gen_str)
        elif decl_conj == '5':
            res.append(stems[0] + "es")
            if len(stems) > 1: res.append(stems[1] + "ei" + gen_str)
        else:
            res.append(stems[0])
            
    elif pos == 'V':
        if decl_conj == '1':
            res.append(stems[0] + "o")
            if len(stems) > 1: res.append(stems[1] + "are")
            if len(stems) > 2: res.append(stems[2] + "i")
            if len(stems) > 3: res.append(stems[3] + "us")
        elif decl_conj == '2':
            res.append(stems[0] + "eo")
            if len(stems) > 1: res.append(stems[1] + "ere")
            if len(stems) > 2: res.append(stems[2] + "i")
            if len(stems) > 3: res.append(stems[3] + "us")
        elif decl_conj == '3':
            # io verbs var? Whitaker uses 3 1 for rego, 3 4 or something for capio? Let's check DB.
            res.append(stems[0] + "o") # approximation
            if len(stems) > 1: res.append(stems[1] + "ere")
            if len(stems) > 2: res.append(stems[2] + "i")
            if len(stems) > 3: res.append(stems[3] + "us")
        elif decl_conj == '4':
            res.append(stems[0] + "io")
            if len(stems) > 1: res.append(stems[1] + "ire")
            if len(stems) > 2: res.append(stems[2] + "i")
            if len(stems) > 3: res.append(stems[3] + "us")
        elif decl_conj == '8': # irregular
            res = stems
        else:
            res = stems
    elif pos == 'ADJ':
        if decl_conj == '1':
            res.append(stems[0] + "us")
            if len(stems) > 1: res.append(stems[1] + "a")
            if len(stems) > 2: res.append(stems[2] + "um")
        elif decl_conj == '3':
            res.append(stems[0] + "is")
            if len(stems) > 1: res.append(stems[1] + "e")
        else:
            res = stems
    else:
        res = stems
        
    return ", ".join(res) if res else stems[0]

import sqlite3
conn = sqlite3.connect('latin_reader/vocab_data/glosses.db')
cur = conn.cursor()
for lemma in ["revoco", "amicus", "capio", "video", "rex", "bonus", "acer", "sum", "qui"]:
    cur.execute("SELECT stem1, stem2, stem3, stem4, pos, morph FROM dictionary WHERE stem1 LIKE ? LIMIT 1", (lemma[:4]+"%",))
    r = cur.fetchone()
    if r:
        stems = list(r[:4])
        print(f"{lemma}: {reconstruct_entry(stems, r[4], r[5])}")
