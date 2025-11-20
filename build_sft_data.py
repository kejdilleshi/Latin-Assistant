# build_sft_data.py
# ------------------------------------------------------------
# Prepare & segment first N (default 20) paragraphs;
# Teacher pass per sentence to generate SFT-ready items.
# Saves:
#   outdir/segments.jsonl
#   outdir/sft_items.jsonl
#
# Requires:
#   pip install openai>=1.40.0 pydantic>=2.5 tqdm
#   export OPENAI_API_KEY=...
#
# Default teacher model: gpt-4o-2024-08-06 (Structured Outputs)
# You can change with --model.
# ------------------------------------------------------------

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional,Tuple

from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# --------------------------
# Pydantic schemas (strict)
# --------------------------

class ForeignSpan(BaseModel):
    lang: str = Field(description="IETF language tag, e.g., 'grc'")
    text: str = Field(description="Foreign (non-Latin) text span, copied verbatim")

class SegmentationResult(BaseModel):
    orthography_policy: str = Field(
        description="Short note of orthographic normalization policy, e.g. 'u->v; keep i/j'"
    )
    normalized_paragraph: str = Field(
        description="Lightly normalized paragraph (do not translate); Greek wrapped as <GRC>…</GRC>"
    )
    sentences: List[str] = Field(
        description="List of sentence strings in normalized orthography with Greek preserved"
    )
    foreign_spans: List[ForeignSpan] = Field(
        default_factory=list,
        description="List of foreign spans mentioned in the paragraph"
    )

class Token(BaseModel):
    form: str
    lemma: str
    upos: str
    feats: Optional[str] = ""     # UD-style: Case=Nom|Gender=Fem|Number=Sing
    head: int = Field(description="1-based head index; 0 = root")
    deprel: str

class TranslationPack(BaseModel):
    literal_en: str
    idiomatic_en: str

class TransformItem(BaseModel):
    type: str = Field(description="e.g., 'Active↔Passive', 'Simplify-ResultClause', 'Singular↔Plural', 'Past↔Future'")
    source_latin: str
    target_latin: str

class MainClause(BaseModel):
    clause: str = Field(description="The identified main clause from the sentence")

class ComplementInfo(BaseModel):
    has_complements: bool
    complements: List[str] = Field(default_factory=list, description="List of identified complements, empty if none")

class TeacherPack(BaseModel):
    translation: TranslationPack
    morphosyntax: List[Token]
    transforms: List[TransformItem]
    main_clause: MainClause
    complement_info: ComplementInfo

# --------------------------
# OpenAI client & helpers
# --------------------------

def new_client():
    try:
        return OpenAI()
    except Exception as e:
        print("Failed to initialize OpenAI client. Did you set OPENAI_API_KEY?", file=sys.stderr)
        raise

def backoff_sleep(attempt: int):
    # Exponential backoff with jitter
    time.sleep(min(60, (2 ** attempt)) + (0.1 * attempt))

def responses_parse_with_backoff(client, *, model: str, sys_msg: str, user_msg: str, text_format):
    attempt = 0
    while True:
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": user_msg},
                ],
                text_format=text_format,
            )
            return resp.output_parsed
        except Exception as e:
            attempt += 1
            if attempt >= 6:
                raise
            backoff_sleep(attempt)

# --------------------------
# Prompts (Segment + Teach)
# --------------------------

SEGMENT_SYSTEM = (
    "You are a Latin philologist and text segmenter. "
    "Return ONLY a valid JSON object conforming to the provided schema. "
    "Do not translate or explain. No commentary."
)

SEGMENT_USER_TMPL = """\
Segment the following Latin paragraph into sentences and lightly normalize orthography.

Normalization policy:
- Replace 'u' with 'v' in Latin words where appropriate (e.g., 'diuisio'→'divisio').
- Keep i/j as-is.
- Preserve all content; do NOT introduce new words.
- Wrap any Greek or clearly non-Latin spans with <GRC>…</GRC> (copy characters unchanged).
- Keep punctuation minimal and standardize to modern dots/commas if needed.

Return fields:
- orthography_policy: exactly "u->v; keep i/j"
- normalized_paragraph: the whole paragraph after the light normalization
- sentences: array of sentence strings (use the normalized text)
- foreign_spans: array of objects {{lang, text}} for each non-Latin span (e.g., lang='grc')

Paragraph (verbatim):
<<<
{paragraph}
>>>
"""

TEACH_SYSTEM = (
    "You are a Classical Latin expert for translation and morpho-syntax. "
    "Produce STRICT JSON matching the schema. No explanations beyond short labels. "
    "Do NOT include chain-of-thought."
)

TEACH_USER_TMPL = """\
For the Latin sentence below, produce a compact teacher pack with:
- translation: literal_en and idiomatic_en (idiomatic should read as natural, fluent English)
- morphosyntax: UD-style tokens (form, lemma, upos, feats, head, deprel). Head is 1-based; 0 for root.
- transforms: 1–3 varied rewrites with increased diversity. Include when applicable:
  * Voice transformations (Active↔Passive)
  * Number transformations (Singular↔Plural)
  * Tense transformations (Past↔Future, where grammatically possible)
  * Clause simplifications or other syntactic rewrites
- main_clause: Identify the main clause of the sentence
- complement_info: Determine if complements are present. If yes, list them; if no, indicate none identified.

Rules:
- Keep Greek text inside <GRC>…</GRC> unchanged.
- Idiomatic translation should read as natural, fluent English.
- Morphological features use UD features (e.g., Case=Gen|Number=Sing|Gender=Neut).
- For transforms, ensure variety and only include those that are grammatically valid for the sentence.
- Keep everything concise.

Latin sentence:
<<<
{sentence}
>>>
"""

# --------------------------
# SFT item builder
# --------------------------

def tokens_to_tabular(tokens: List[Token]) -> str:
    lines = []
    for t in tokens:
        feats = t.feats if (t.feats is not None and t.feats != "") else "_"
        lines.append(
            f"{t.form}\t{t.lemma}\t{t.upos}\t{feats}\t{t.head}\t{t.deprel}"
        )
    return "\n".join(lines)

def make_sft_items(
    paragraph_idx: int,
    sentence_idx: int,
    sentence_text: str,
    pack: TeacherPack,
    max_transforms: int = 3,
):
    base_meta = {
        "source": {"paragraph_index": paragraph_idx, "sentence_index": sentence_idx},
    }
    pid = f"p{paragraph_idx:04d}.s{sentence_idx:02d}"

    items = []

    # 1) Translation (idiomatic only)
    items.append({
        "id": f"{pid}.t.translate",
        "task": "translate",
        "prompt": f"Task: Translate the Latin sentence into idiomatic English. Latin: {sentence_text}",
        "target": pack.translation.idiomatic_en.strip(),
        "meta": base_meta,
    })

    # 2) Morpho-syntax (tabular)
    items.append({
        "id": f"{pid}.a.morphosyntax",
        "task": "morphosyntax",
        "prompt": (
            "Task: For each token, output FORM<TAB>LEMMA<TAB>UPOS<TAB>FEATS<TAB>HEAD<TAB>DEPREL "
            f"for the sentence. Latin: {sentence_text}"
        ),
        "target": tokens_to_tabular(pack.morphosyntax),
        "meta": base_meta,
    })

    # 3) Transforms (with increased variety)
    for i, tr in enumerate(pack.transforms[:max_transforms], start=1):
        items.append({
            "id": f"{pid}.x.{i:02d}",
            "task": "transform",
            "prompt": f"Task: Rewrite in Latin as instructed ({tr.type}). Input: {tr.source_latin}",
            "target": tr.target_latin.strip(),
            "meta": {**base_meta, "transform": tr.type},
        })

    # 4) Main Clause Identification
    items.append({
        "id": f"{pid}.c.main_clause",
        "task": "main_clause",
        "prompt": f'Task: In this text "{sentence_text}", what is the main clause?',
        "target": pack.main_clause.clause.strip(),
        "meta": base_meta,
    })

    # 5) Complement Identification
    if pack.complement_info.has_complements:
        complement_list = ", ".join(pack.complement_info.complements)
        target = f"Yes. Complements identified: {complement_list}"
    else:
        target = "No complements are identified."

    items.append({
        "id": f"{pid}.c.complements",
        "task": "complement_identification",
        "prompt": f"Task: Does this text contain any complements? If so, which ones? Latin: {sentence_text}",
        "target": target,
        "meta": base_meta,
    })

    return items

# --------------------------
# Dataset loading
# --------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj and isinstance(obj["text"], str):
                text = obj["text"]
            else:
                # Fallback: best-effort attempt to find a string field
                text = None
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, str):
                            text = v
                            break
                if text is None:
                    raise ValueError(f"JSONL line {i} lacks a string 'text' field.")
            records.append({"id": obj.get("id", f"rec{i:04d}"), "text": text})
    return records

def split_numbered_paragraphs(raw: str) -> List[Tuple[int, str]]:
    """
    Split a text file into (numeric_id, paragraph_text) pairs.
    A paragraph starts at any line that begins with: digits '.' space?
      e.g., '1. ', '12. ', '8.'.

    Collect all subsequent lines until the next numbered start or EOF.
    The numeric label is stripped from the paragraph text.
    """
    blocks: List[Tuple[int, str]] = []
    current_id: int = None  # type: ignore
    current_lines: List[str] = []

    lines = raw.replace("\r\n", "\n").split("\n")
    start_pat = re.compile(r"^\s*(\d+)\.\s*(.*)$")

    for line in lines:
        m = start_pat.match(line)
        if m:
            # start a new paragraph
            if current_id is not None:
                paragraph_text = "\n".join(current_lines).strip()
                if paragraph_text:
                    blocks.append((current_id, paragraph_text))
            current_id = int(m.group(1))
            tail = m.group(2)
            current_lines = [tail] if tail else []
        else:
            # accumulate lines within the current paragraph (including blank lines)
            if current_id is not None:
                current_lines.append(line)

    # flush last
    if current_id is not None:
        paragraph_text = "\n".join(current_lines).strip()
        if paragraph_text:
            blocks.append((current_id, paragraph_text))

    return blocks

def load_txt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    numbered = split_numbered_paragraphs(raw)
    if numbered:
        # Use the numeric label as a stable id
        return [{"id": f"rec{pid:04d}", "text": text} for pid, text in numbered]

    # Fallback: blank-line-separated paragraphs
    blocks = re.split(r"\n\s*\n+", raw.strip())
    blocks = [b.strip() for b in blocks if b.strip()]
    return [{"id": f"rec{i:04d}", "text": p} for i, p in enumerate(blocks, start=1)]

def load_records(path: str) -> List[dict]:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".jsonl":
        return load_jsonl(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError("Unsupported input format. Use .jsonl (with 'text' field) or .txt (blank-line separated).")

# --------------------------
# Pipeline
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Build SFT data for Mistral-7B from Latin paragraphs via teacher model.")
    ap.add_argument("input_path", type=str, default='data/latin_corpus.txt',help="Path to dataset (.jsonl with 'text' or .txt paragraphs)")
    ap.add_argument("--outdir", type=str, default="out_sft+", help="Output directory")
    ap.add_argument("--max_items", type=int, default=-1, help="Number of paragraphs to process")
    ap.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Teacher model (Responses API)")
    ap.add_argument("--max_transforms", type=int, default=3, help="Max transforms per sentence")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seg_path = os.path.join(args.outdir, "segments.jsonl")
    sft_path = os.path.join(args.outdir, "sft_items.jsonl")

    client = new_client()
    records = load_records(args.input_path)
    subset = records[: args.max_items]

    # Open output files
    seg_out = open(seg_path, "w", encoding="utf-8")
    sft_out = open(sft_path, "w", encoding="utf-8")

    total_sent = 0
    total_items = 0

    try:
        pbar = tqdm(enumerate(subset, start=1), total=len(subset), desc="Processing paragraphs", unit="para")
        for p_idx, rec in pbar:
            paragraph = rec["text"]
            seg_user = SEGMENT_USER_TMPL.format(paragraph=paragraph)

            # 1) SEGMENTATION
            seg = responses_parse_with_backoff(
                client, model=args.model, sys_msg=SEGMENT_SYSTEM, user_msg=seg_user, text_format=SegmentationResult
            )
            # Validate explicitly (defensive)
            try:
                seg = SegmentationResult.model_validate(seg)
            except ValidationError as ve:
                tqdm.write(f"[WARN] Segmentation validation failed for paragraph {p_idx}: {ve}")
                continue

            seg_record = {
                "id": rec["id"],
                "paragraph_index": p_idx,
                "orthography_policy": seg.orthography_policy,
                "normalized_paragraph": seg.normalized_paragraph,
                "sentences": seg.sentences,
                "foreign_spans": [fs.model_dump() for fs in seg.foreign_spans],
            }
            seg_out.write(json.dumps(seg_record, ensure_ascii=False) + "\n")
            seg_out.flush()

            # 2) TEACHER PASS per sentence
            for s_idx, sent in enumerate(seg.sentences, start=1):
                total_sent += 1
                teach_user = TEACH_USER_TMPL.format(sentence=sent)

                try:
                    pack = responses_parse_with_backoff(
                        client, model=args.model, sys_msg=TEACH_SYSTEM, user_msg=teach_user, text_format=TeacherPack
                    )
                    pack = TeacherPack.model_validate(pack)
                except Exception as e:
                    tqdm.write(f"[WARN] Teacher pack failed at p={p_idx}, s={s_idx}: {e}")
                    continue

                items = make_sft_items(
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                    sentence_text=sent,
                    pack=pack,
                    max_transforms=args.max_transforms,
                )
                for it in items:
                    sft_out.write(json.dumps(it, ensure_ascii=False) + "\n")
                    total_items += 1
                sft_out.flush()

                # Update progress bar with current stats
                pbar.set_postfix({
                    'sentences': total_sent,
                    'items': total_items
                })

        print(f"Done. Paragraphs processed: {len(subset)}")
        print(f"Sentences processed: {total_sent}")
        print(f"SFT items written: {total_items}")
        print(f"- Segments: {seg_path}")
        print(f"- SFT items: {sft_path}")

    finally:
        seg_out.close()
        sft_out.close()

if __name__ == "__main__":
    main()
