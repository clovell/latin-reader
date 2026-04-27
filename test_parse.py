def parse(stems, pos, morph):
    decl_conj = morph[0:3].strip()
    # just an example
    print(f"Stems: {stems}, POS: {pos}, Morph: {decl_conj}")

parse(['revoc', 'revoc', 'revocav', 'revocat'], 'V', '1 1 X X X')
parse(['amic', 'amic', 'zzz', 'zzz'], 'N', '2 1 M T')
