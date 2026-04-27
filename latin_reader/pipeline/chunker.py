"""Chunker module (Phase 3).

Converts a flat dependency tree into a nested tree of Chunk objects.
Uses a best-effort heuristic approach, falling back to a flat list on errors.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional, Set

from latin_reader.latincy_postprocessor.sentence import Token

logger = logging.getLogger(__name__)

ChunkType = Literal[
    "Sentence", "Clause", "MainClause", "SubordinateClause",
    "RelativeClause", "AblativeAbsolute", "IndirectStatement",
    "NP", "PP", "VP", "ParticipialPhrase",
    "AdjP", "AdvP", "Coordination", "Token",
]


@dataclass
class Chunk:
    type: ChunkType
    label: str
    token_ids: list[int]
    head_token_id: int | None = None
    children: list["Chunk"] = field(default_factory=list)
    role: str | None = None
    notes: list[str] = field(default_factory=list)

    def flatten_token_ids(self) -> list[int]:
        """Return all token IDs in this chunk and its descendants, sorted."""
        ids = set(self.token_ids)
        for child in self.children:
            ids.update(child.flatten_token_ids())
        return sorted(list(ids))


class SentenceGraph:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.tok_map: Dict[int, Token] = {t.id: t for t in tokens}
        self.children: Dict[int, List[Token]] = {t.id: [] for t in tokens}
        self.children[0] = []  # For root
        
        for t in tokens:
            if t.head in self.children:
                self.children[t.head].append(t)

        self._processed_ids: Set[int] = set()

    def get_children(self, token_id: int) -> List[Token]:
        return self.children.get(token_id, [])


def _get_clause_heads(graph: SentenceGraph) -> List[Token]:
    """Identify tokens that serve as clause heads."""
    clause_deprels = {
        "root", "advcl", "ccomp", "conj"
    }
    heads = []
    for t in graph.tokens:
        deprel_base = t.deprel.lower().split(":")[0]  # e.g., 'advcl:abs' -> 'advcl'
        # Treat conjunctions as clauses if they link verbs or clauses.
        # simple heuristic: 'conj' with a verb form.
        if deprel_base in clause_deprels:
            if deprel_base == "conj" and t.upos not in ("VERB", "AUX") and t.deprel.lower() != "conj:clause":
                # Likely a noun/adj coordination, not a clausal one. This is a heuristic.
                continue
            heads.append(t)
    # Ensure there's at least one root if possible
    if not heads and graph.children.get(0):
        heads.append(graph.children[0][0])
    return heads


def _build_np(head_tok: Token, graph: SentenceGraph) -> Chunk:
    """Build a Noun Phrase chunk."""
    np_deps = {"amod", "det", "nummod", "nmod", "punct", "case", "cc", "advmod"}
    children = graph.get_children(head_tok.id)
    
    chunk_children = []
    token_ids = [head_tok.id]
    
    # We don't recursively build full sub-chunks for every modifier here in v1,
    # except collecting the direct simple modifiers.
    # To be safer with recursion, we'll just gather the direct tokens.
    for child in children:
        deprel_base = child.deprel.split(":")[0]
        if deprel_base in np_deps and child.id not in graph._processed_ids:
            graph._processed_ids.add(child.id)
            # if the modifier has its own children, they might get dropped unless we recurse.
            # But the spec says "group NOUN + amod/det...". Let's wrap the child as a Token chunk
            # or sub-NP. For now, flat tokens in the NP are assigned to token_ids.
            token_ids.append(child.id)
            
            # Recurse for the child's children just to gather their ids (e.g. adverbs modifying adjectives)
            # A simple way is to capture the subtree rooted at child if it's purely modifiers
            def _gather_subtree(tok):
                for gc in graph.get_children(tok.id):
                    # avoiding complex subclauses
                    deprel_base = gc.deprel.split(":")[0]
                    if deprel_base in {"acl", "rcmod", "advcl"}:
                        continue
                    # Include word-level conj and cc (e.g. "apsenti et praesenti")
                    # but skip clausal conjunctions
                    if deprel_base == "conj" and gc.upos in ("VERB", "AUX"):
                        continue
                    token_ids.append(gc.id)
                    graph._processed_ids.add(gc.id)
                    _gather_subtree(gc)
            _gather_subtree(child)

    token_ids = sorted(list(set(token_ids)))
    return Chunk(
        type="NP",
        label=f"NP ({head_tok.form})",
        token_ids=token_ids,
        head_token_id=head_tok.id,
        role=head_tok.deprel
    )

def _build_pp(head_tok: Token, case_tok: Token, graph: SentenceGraph) -> Chunk:
    """Build a Prepositional Phrase chunk."""
    np_chunk = _build_np(head_tok, graph)
    token_ids = sorted([case_tok.id] + np_chunk.flatten_token_ids())
    return Chunk(
        type="PP",
        label=f"PP ({case_tok.form} {head_tok.form})",
        token_ids=token_ids,
        head_token_id=case_tok.id,  # Preposition is head of PP
        role=head_tok.deprel,
        children=[np_chunk] if np_chunk.token_ids != [head_tok.id] else [] # simplify if no modifiers
    )

def _build_participial_phrase(head_tok: Token, graph: SentenceGraph) -> Chunk:
    """Build a phrase around a participle."""
    token_ids = [head_tok.id]
    chunk_children = []
    for child in graph.get_children(head_tok.id):
        if child.id not in graph._processed_ids and child.deprel.split(":")[0] not in {"conj"}:
            graph._processed_ids.add(child.id)
            sub_chunk = _build_component(child, graph)
            if sub_chunk:
                chunk_children.append(sub_chunk)
    
    return Chunk(
        type="ParticipialPhrase",
        label=f"PartPhrase ({head_tok.form})",
        token_ids=token_ids,
        head_token_id=head_tok.id,
        children=chunk_children,
        role=head_tok.deprel
    )

def _build_coordination(head_tok: Token, graph: SentenceGraph) -> Chunk:
    """Build a coordination wrapper."""
    # head_tok is typically the first conjunct
    token_ids = [head_tok.id]
    chunk_children = []
    
    # Process head's non-conj children as its own group first
    head_component_children = []
    case_tok = None
    for child in graph.get_children(head_tok.id):
        if child.deprel == "case" and child.id not in graph._processed_ids:
            case_tok = child
            graph._processed_ids.add(child.id)
        elif child.deprel != "conj" and child.deprel != "cc" and child.id not in graph._processed_ids:
            graph._processed_ids.add(child.id)
            sub = _build_component(child, graph)
            if sub: head_component_children.append(sub)
    
    # Construct the first conjunct
    first_conjunct = Chunk(
        type="NP" if head_tok.upos in ("NOUN", "PROPN", "PRON") else "VP",
        label=f"Conjunct ({head_tok.form})",
        token_ids=[head_tok.id],
        head_token_id=head_tok.id,
        children=head_component_children
    )
    chunk_children.append(first_conjunct)
    
    # Process cc and conj children
    for child in graph.get_children(head_tok.id):
        if child.id not in graph._processed_ids:
            if child.deprel == "cc":
                graph._processed_ids.add(child.id)
                token_ids.append(child.id)  # CC is part of the Coord wrapper
            elif child.deprel == "conj":
                graph._processed_ids.add(child.id)
                conj_chunk = _build_component(child, graph)
                if conj_chunk:
                    conj_chunk.label = f"Conjunct ({child.form})"
                    chunk_children.append(conj_chunk)

    coord_chunk = Chunk(
        type="Coordination",
        label="Coordination",
        token_ids=sorted(token_ids),
        head_token_id=head_tok.id,
        children=chunk_children,
        role=head_tok.deprel
    )
    
    if case_tok:
        return Chunk(
            type="PP",
            label=f"PP ({case_tok.form} ...)",
            token_ids=sorted([case_tok.id] + coord_chunk.flatten_token_ids()),
            head_token_id=case_tok.id,
            role=head_tok.deprel,
            children=[coord_chunk]
        )
    return coord_chunk

def _build_component(tok: Token, graph: SentenceGraph) -> Optional[Chunk]:
    """Recursively process a token and its dependents based on POS/deprel."""
    
    # 0. Check if this component is actually a clause!
    clause_deprels = {"advcl", "ccomp", "conj"}
    deprel_base = tok.deprel.split(":")[0]
    if deprel_base in clause_deprels:
        if deprel_base == "conj" and tok.upos not in ("VERB", "AUX") and tok.deprel != "conj:clause":
            pass # Simple word-level conjunction, let it fall through
        else:
            return _build_clause(tok, graph)
        
    children = graph.get_children(tok.id)
    
    # 1. Check for coordination (unless we are already processing a conjunct)
    # A token with 'conj' children is a coordination head.
    has_conj = any(c.deprel == "conj" for c in children)
    if has_conj and tok.deprel != "conj":
         return _build_coordination(tok, graph)

    # 2. Check for PPs
    case_tok = next((c for c in children if c.deprel == "case"), None)
    if case_tok and tok.upos in ("NOUN", "PROPN", "PRON") and case_tok.id not in graph._processed_ids:
        graph._processed_ids.add(case_tok.id)
        return _build_pp(tok, case_tok, graph)

    # 3. Check for NPs
    if tok.upos in ("NOUN", "PROPN", "PRON"):
        return _build_np(tok, graph)

    # 4. Check for Participles
    if tok.upos == "VERB" and "VerbForm=Part" in getattr(tok, "feats", {}).get("VerbForm", ""):
         # Note: in CoNLL feats are usually serialized or we can check the string
         return _build_participial_phrase(tok, graph)
    if tok.upos == "VERB" and hasattr(tok, 'feats') and isinstance(tok.feats, dict) and tok.feats.get("VerbForm") == "Part":
        return _build_participial_phrase(tok, graph)

    # 5. Default single generic token chunk with potential children
    chunk_children = []
    token_ids = [tok.id]
    for child in children:
        if child.id not in graph._processed_ids:
            # specifically for mark, aux, cop, punct - keep them flat with head
            if child.deprel in ("mark", "aux", "cop", "punct", "cc", "case"):
                graph._processed_ids.add(child.id)
                token_ids.append(child.id)
            else:
                graph._processed_ids.add(child.id)
                sub_chunk = _build_component(child, graph)
                if sub_chunk:
                    chunk_children.append(sub_chunk)

    # determine basic generic label
    t_type = "VP" if tok.upos in ("VERB", "AUX") else ("AdjP" if tok.upos == "ADJ" else ("AdvP" if tok.upos == "ADV" else "Token"))
    
    chunk = Chunk(
        type=t_type,  # type: ignore
        label=f"{t_type} ({tok.form})",
        token_ids=sorted(token_ids),
        head_token_id=tok.id,
        children=chunk_children,
        role=tok.deprel
    )
    return chunk


def _build_clause(head_tok: Token, graph: SentenceGraph) -> Chunk:
    """Build a clause chunk."""
    children = graph.get_children(head_tok.id)
    
    # Determine clause type
    clause_type: ChunkType = "Clause"
    label = f"Clause ({head_tok.form})"
    
    if head_tok.deprel == "root":
        clause_type = "MainClause"
        label = f"Main Clause ({head_tok.form})"
    elif head_tok.deprel == "advcl:abs":
        clause_type = "AblativeAbsolute"
        label = f"Ablative Absolute ({head_tok.form})"
    elif head_tok.deprel in ("ccomp", "xcomp"):
        # Check for accusative subject -> indirect statement
        has_acc_subj = any(c.deprel.startswith("nsubj") and c.feats.get("Case") == "Acc" for c in children)
        is_inf = head_tok.feats.get("VerbForm") == "Inf"
        if is_inf and has_acc_subj:
            clause_type = "IndirectStatement"
            label = f"Indirect Statement ({head_tok.form})"
        else:
            clause_type = "SubordinateClause"
            label = f"Comp Clause ({head_tok.form})"
    elif head_tok.deprel.startswith("acl"):
        clause_type = "RelativeClause" if head_tok.deprel == "acl:relcl" else "SubordinateClause"
        label = f"Relative Clause ({head_tok.form})" if clause_type == "RelativeClause" else f"Subordinate Clause ({head_tok.form})"
    elif any(c.deprel == "mark" for c in children) or head_tok.deprel == "advcl":
        clause_type = "SubordinateClause"
        label = f"Subordinate Clause ({head_tok.form})"
    
    token_ids = [head_tok.id]
    chunk_children = []
    
    # mark markers, aux, cop, punct as belonging directly to this clause token_ids
    for child in children:
        if child.id not in graph._processed_ids:
            if child.deprel in ("mark", "aux", "cop", "punct") or child.deprel.startswith("aux:"):
                graph._processed_ids.add(child.id)
                token_ids.append(child.id)
            else:
                graph._processed_ids.add(child.id)
                sub_chunk = _build_component(child, graph)
                if sub_chunk:
                    chunk_children.append(sub_chunk)
                    
    return Chunk(
        type=clause_type,
        label=label,
        token_ids=sorted(token_ids),
        head_token_id=head_tok.id,
        children=chunk_children,
        role=head_tok.deprel
    )

def chunk_sentence(tokens: List[Token]) -> Chunk:
    """Safely convert a sentence of tokens into a nested Chunk tree."""
    try:
        # Quick validation
        if not tokens:
             return Chunk(type="Sentence", label="Empty Sentence", token_ids=[])

        graph = SentenceGraph(tokens)
        clause_heads = _get_clause_heads(graph)
        
        # If the heuristic fails to find the main architecture, fallback.
        if not clause_heads:
            raise ValueError("No clause heads found.")
            
        sentence_children = []
        
        for head in clause_heads:
            if head.id not in graph._processed_ids:
                graph._processed_ids.add(head.id)
                clause_chunk = _build_clause(head, graph)
                sentence_children.append(clause_chunk)

        # Catch any orphans
        unprocessed = [t for t in tokens if t.id not in graph._processed_ids and t.id not in [c for c in [ids for chunk in sentence_children for ids in chunk.flatten_token_ids()]]]
        for t in unprocessed:
            graph._processed_ids.add(t.id)
            orphan_chunk = _build_component(t, graph)
            if orphan_chunk:
                sentence_children.append(orphan_chunk)

        root_chunk = Chunk(
            type="Sentence",
            label="Sentence",
            token_ids=[],
            head_token_id=None,
            children=sentence_children
        )
        
        # --- Typographical Punctuation Reparenting ---
        # To prevent punctuation from gap-jumping across extracted subclauses,
        # we forcefully re-assign every punctuation mark to the chunk containing its strict linear predecessor.
        tok_map = {t.id: t for t in tokens}
        
        def _remove_punct(c: Chunk):
            c.token_ids = [tid for tid in c.token_ids if tok_map[tid].upos != "PUNCT"]
            for child in c.children:
                _remove_punct(child)
                
        def _build_tid_map(c: Chunk, d: dict):
            for tid in c.token_ids:
                d[tid] = c
            for child in c.children:
                _build_tid_map(child, d)
                
        _remove_punct(root_chunk)
        
        tid_to_chunk = {}
        _build_tid_map(root_chunk, tid_to_chunk)
        
        # Any tokens that were just orphaned outright are safe to ignore, but we must place Punct.
        for t in tokens:
            if t.upos == "PUNCT" or t.form in (",", ".", ";", ":", "?", "!", '"', "'"):
                # Find closest preceding word
                target_id = t.id - 1
                while target_id > 0 and (tok_map[target_id].upos == "PUNCT" or tok_map[target_id].form in (",", ".", ";", ":", "?", "!", '"', "'")):
                    target_id -= 1
                    
                if target_id > 0 and target_id in tid_to_chunk:
                    tid_to_chunk[target_id].token_ids.append(t.id)
                    tid_to_chunk[target_id].token_ids.sort()
                else:
                    root_chunk.token_ids.append(t.id)
                    root_chunk.token_ids.sort()

        return root_chunk

    except Exception as e:
        logger.warning(f"Chunker failed: {e}. Falling back to flat representation.")
        # Defensive fallback: return a flat list of Token chunks
        fallback_children = []
        for t in tokens:
            fallback_children.append(Chunk(
                type="Token",
                label=t.form,
                token_ids=[t.id],
                head_token_id=t.id,
                role=t.deprel
            ))
        return Chunk(
            type="Sentence",
            label="Sentence (Flat Fallback)",
            token_ids=[],
            head_token_id=None,
            children=fallback_children,
            notes=["Chunking failed, showing flat structure."]
        )
