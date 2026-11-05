#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quranic Hybrid Lexical-Semantic Retrieval | Consolidated Research Code
====================================================================

This ONE file combines the four supplied source files:
    - preProcessing.py
    - data_preparation.ipynb
    - 02_feature_extraction.ipynb
    - Feature Enginnering.ipynb

No machine-specific paths, usernames, email addresses, or credentials are used.
The code uses project-relative folders, which may be overridden on the CLI.

Suggested project layout:
    Quran_Hybrid_Retrieval_All_In_One.py
    data/
        quran-simple.txt                   # verse lines: sura|ayah|text
        ibn_kathir_similarity.csv           # optional for pair-based stages
        QConcept.json                       # optional
        QAnaphor.json                       # optional
        QacSegment.json                     # optional
        GrammarTag.json                     # optional
    output/                                 # generated automatically

Install the packages needed for the stages you will run:
    pip install pandas numpy scipy scikit-learn nltk camel-tools
    pip install torch transformers sentence-transformers tqdm matplotlib
    pip install rank-bm25                  # optional BM25 baseline

Examples (from the folder containing this script):
    python Quran_Hybrid_Retrieval_All_In_One.py preprocess
    python Quran_Hybrid_Retrieval_All_In_One.py prepare-pairs
    python Quran_Hybrid_Retrieval_All_In_One.py pair-similarity
    python Quran_Hybrid_Retrieval_All_In_One.py pair-charts
    python Quran_Hybrid_Retrieval_All_In_One.py features
    python Quran_Hybrid_Retrieval_All_In_One.py search --query "الحمد لله رب العالمين"
    python Quran_Hybrid_Retrieval_All_In_One.py evaluate --protocol original
    python Quran_Hybrid_Retrieval_All_In_One.py evaluate --protocol shared
    python Quran_Hybrid_Retrieval_All_In_One.py alpha-sweep
    python Quran_Hybrid_Retrieval_All_In_One.py bm25 --query "اهدنا الصراط المستقيم"
    python Quran_Hybrid_Retrieval_All_In_One.py pipeline

The feature-engineering path loads PRETRAINED AraBERTv2 through
SentenceTransformers and does not fine-tune it. A separate original notebook
used the differently named AraBERT v02 checkpoint for verse-pair analysis.
The two paths are preserved as separate experimental stages.

IMPORTANT EVALUATION NOTE:
    --protocol original reproduces the notebook's comparison DESIGN:
    different queries for lexical/semantic/hybrid and mean RR over all
    expected verses. That number is NOT standard MRR, and these figures
    should not be used as a fair cross-model benchmark.
    --protocol shared applies the SAME query sets to the three modes and
    computes the usual first-relevant reciprocal rank. The original manually
    selected relevance lists are still limited and not independently verified.
    No previously reported paper metrics are assumed or hardcoded as results.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# 01 | PROJECT SETTINGS AND PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

# The exact identifiers used in the two supplied notebook pipelines.
RETRIEVAL_MODEL = "aubmindlab/bert-base-arabertv2"
PAIR_ANALYSIS_MODEL = "aubmindlab/bert-base-arabertv02"
DEFAULT_ALPHA = 0.70  # 70% semantic + 30% lexical (original main model)

ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "أن", "إن", "كان", "ما", "لا", "لم",
    "لن", "قد", "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن",
    "انا", "نحن", "انت", "انتم", "الذي", "التي", "اللذين", "اللذان",
    "اللائي", "اللاتي", "ثم", "أو", "بل", "و", "ف", "ب", "ك", "ل",
    "حتى", "أيضا", "اين", "متى", "كيف", "هناك",
}


def dirs(data_dir: Path, output_dir: Path) -> dict[str, Path]:
    paths = {
        "data": Path(data_dir).expanduser().resolve(),
        "processed": Path(output_dir).expanduser().resolve() / "processed",
        "features": Path(output_dir).expanduser().resolve() / "features",
        "results": Path(output_dir).expanduser().resolve() / "results",
    }
    for key in ("processed", "features", "results"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}. See the input layout in --help.")
    return path


def dump_pickle(object_to_save: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    # Use only files you created or trust. Pickle can execute malicious code.
    with require_file(path).open("rb") as handle:
        return pickle.load(handle)


# ============================================================================
# 02 | QURAN TEXT CLEANING AND PREPROCESSING (preProcessing.py)
# ============================================================================


def clean_text(text: Any) -> str:
    text = str(text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", text)
    text = re.sub(r"[۞۩﴿﴾ـ]", "", text)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_arabic(text: str) -> str:
    # Preserve the original notebook's normalization rules.
    text = re.sub("[إأآٱا]", "ا", text)
    for before, after in (
        ("ى", "ي"), ("ؤ", "و"), ("ئ", "ي"), ("ة", "ه"),
        ("گ", "ك"), ("پ", "ب"), ("چ", "ج"), ("ژ", "ز"), ("ڤ", "ف"),
    ):
        text = text.replace(before, after)
    return text


def preprocess_for_tfidf(text: Any) -> str:
    # The query side of the original notebook used normalized, cleaned text.
    return normalize_arabic(clean_text(text))


def tokenize_words(text: str) -> list[str]:
    """Prefer the original CAMeL tokenizer; cleaned Arabic is whitespace-only."""
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        return simple_word_tokenize(text)
    except ImportError:
        # After clean_text(), all punctuation is removed, so space splitting
        # is equivalent for the cleaned verse strings in this pipeline.
        return text.split()


def load_quran(data_dir: Path) -> pd.DataFrame:
    """Load quran-simple.txt or the original pipe-delimited quran.csv."""
    candidates = [data_dir / "quran-simple.txt", data_dir / "quran.csv"]
    selected = next((p for p in candidates if p.is_file()), None)
    if selected is None:
        raise FileNotFoundError("Place quran-simple.txt (or pipe-delimited quran.csv) in data/.")

    verses: list[tuple[int, int, str]] = []
    with selected.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split("|", 2)
            if len(parts) != 3:
                continue  # matches the original input's text/metadata filtering
            try:
                sura, ayah = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            verses.append((sura, ayah, parts[2].strip()))
    if not verses:
        raise ValueError(f"No verse lines found in {selected}.")
    return (pd.DataFrame(verses, columns=["sura", "ayah", "ayah_text"])
            .drop_duplicates(subset=["sura", "ayah"]).reset_index(drop=True))


def preprocess_quran(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    quran = load_quran(data_dir)
    quran["clean_text"] = quran["ayah_text"].map(preprocess_for_tfidf)
    quran["tokens"] = quran["clean_text"].map(tokenize_words)

    from nltk.stem.isri import ISRIStemmer
    stemmer = ISRIStemmer()
    quran["stems"] = quran["tokens"].map(
        lambda tokens: [stemmer.stem(t) for t in tokens]
    )
    quran["tokens_no_stopwords"] = quran["tokens"].map(
        lambda tokens: [t for t in tokens if t not in ARABIC_STOPWORDS]
    )
    out = processed_dir / "quran_final_preprocessed.csv"
    quran.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(quran)} preprocessed verses -> {out}")
    return quran


def _simple_clean(text: Any) -> str:
    return clean_text(text)


def preprocess_optional_annotations(data_dir: Path, processed_dir: Path) -> None:
    """Clean auxiliary resources from the original .py if supplied.

    These are saved as extra tables; the original retrieval model does NOT
    automatically incorporate these annotation tables into its feature vectors.
    """
    for file_name in ("QConcept.json", "QAnaphor.json", "QacSegment.json", "GrammarTag.json"):
        source = data_dir / file_name
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8-sig") as handle:
            frame = pd.DataFrame(json.load(handle))
        if file_name == "QConcept.json":
            if "Name" in frame.columns:
                frame["clean_name"] = frame["Name"].map(lambda x: normalize_arabic(_simple_clean(x)))
            if "Gloss" in frame.columns:
                frame["clean_gloss"] = frame["Gloss"].map(
                    lambda x: re.sub(r"[()\[\],;]", "", str(x))
                )
        elif file_name == "QAnaphor.json":
            cols = [c for c in frame if c.endswith("Id") or c.startswith("_id")]
            if cols:
                frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            frame = frame.dropna(how="all")
        elif file_name == "QacSegment.json":
            for field in ("Text", "Root", "Lemma"):
                if field in frame.columns:
                    frame[f"{field}_Arabic_Clean"] = frame[field].map(
                        lambda x: normalize_arabic(_simple_clean(x["Arabic"]))
                        if isinstance(x, dict) and "Arabic" in x else ""
                    )
        elif file_name == "GrammarTag.json":
            if "_id" in frame.columns:
                frame["_id"] = pd.to_numeric(frame["_id"], errors="coerce").astype("Int64")
            if "Arabic" in frame.columns:
                frame["Arabic_Clean"] = frame["Arabic"].map(
                    lambda x: re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", str(x)).strip()
                )
            frame = frame.drop_duplicates()
        out = processed_dir / f"{source.stem}_clean.csv"
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved optional annotations -> {out}")


def get_processed_quran(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(processed_dir / "quran_final_preprocessed.csv"))


# ============================================================================
# 03 | VERSE PAIR DATA PREPARATION (data_preparation.ipynb)
# ============================================================================


def prepare_verse_pairs(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    source = require_file(data_dir / "ibn_kathir_similarity.csv")
    relations = pd.read_csv(source)
    id_cols = ["Source_Sura", "Source_Verse", "target_sura", "Target_verse"]
    for col in id_cols:
        relations[col] = pd.to_numeric(relations[col], errors="coerce")
    relations = relations.dropna(subset=id_cols).copy()
    relations[id_cols] = relations[id_cols].astype(int)
    relations = relations.drop_duplicates()
    if "common_roots" not in relations:
        relations["common_roots"] = np.nan
    if "relevance_degree" not in relations:
        relations["relevance_degree"] = np.nan

    verse_text = load_quran(data_dir)[["sura", "ayah", "ayah_text"]]
    merged = relations.merge(
        verse_text, left_on=["Source_Sura", "Source_Verse"],
        right_on=["sura", "ayah"], how="left"
    ).rename(columns={"ayah_text": "source_text"})
    merged = merged.merge(
        verse_text, left_on=["target_sura", "Target_verse"],
        right_on=["sura", "ayah"], how="left", suffixes=("_source", "_target")
    ).rename(columns={"ayah_text": "target_text"})
    merged = merged[[
        "Source_Sura", "Source_Verse", "source_text", "target_sura",
        "Target_verse", "target_text", "common_roots", "relevance_degree",
    ]]
    missing = int(merged[["source_text", "target_text"]].isna().any(axis=1).sum())
    if missing:
        warnings.warn(f"{missing} verse pairs lack a matching source/target text.")
    out = processed_dir / "quran_pairs_merged.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(merged)} verse pairs -> {out}")
    return merged


# ============================================================================
# 04 | ARABERT SIMILARITY FOR IBN KATHIR VERSE PAIRS
#      (02_feature_extraction.ipynb; separate from the search model)
# ============================================================================


def extract_pair_similarities(processed_dir: Path, results_dir: Path, model_name: str = PAIR_ANALYSIS_MODEL) -> pd.DataFrame:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    pairs = pd.read_csv(require_file(processed_dir / "quran_pairs_merged.csv"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # inference only; NOT fine-tuning

    @torch.no_grad()
    def embed(text: str):
        tokens = tokenizer(str(text), return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        hidden = model(**tokens).last_hidden_state
        return hidden.mean(dim=1).squeeze(0)  # original mean-pooling method

    scores: list[float] = []
    for row in pairs.itertuples(index=False):
        if pd.isna(row.source_text) or pd.isna(row.target_text):
            scores.append(float("nan"))
            continue
        one = embed(row.source_text)
        two = embed(row.target_text)
        scores.append(float(F.cosine_similarity(one.unsqueeze(0), two.unsqueeze(0)).item()))
    pairs["arabert_similarity"] = scores
    out = processed_dir / "quran_pairs_with_arabert_similarity.csv"
    pairs.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved pair similarity scores -> {out}")
    return pairs


def pair_analysis_charts(processed_dir: Path, results_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(require_file(processed_dir / "quran_pairs_with_arabert_similarity.csv"))
    valid = df.dropna(subset=["arabert_similarity"]).copy()
    if valid.empty:
        raise ValueError("No numerical AraBERT similarities were found.")
    scaler = MinMaxScaler()
    valid["normalized_similarity"] = scaler.fit_transform(valid[["arabert_similarity"]])
    output = results_dir / "arabert_pair_statistics.csv"
    valid[["arabert_similarity", "normalized_similarity"]].describe().to_csv(output)
    print(f"Saved pair statistics -> {output}")

    if "relevance_degree" in valid:
        valid["relevance_degree"] = pd.to_numeric(valid["relevance_degree"], errors="coerce")
        subset = valid.dropna(subset=["relevance_degree"])
        if len(subset) >= 2:
            corr = subset["arabert_similarity"].corr(subset["relevance_degree"])
            print(f"Correlation with Ibn Kathir relevance degree: {corr:.4f}")
            plt.figure(figsize=(7, 5))
            plt.scatter(subset["relevance_degree"], subset["arabert_similarity"], alpha=.4)
            plt.xlabel("Ibn Kathir relation degree")
            plt.ylabel("AraBERT semantic similarity")
            plt.title("AraBERT Similarity vs Ibn Kathir Relation Degree")
            plt.tight_layout()
            plt.savefig(results_dir / "arabert_scatter.png", dpi=300)
            plt.close()

            grouped = [part["arabert_similarity"].to_numpy() for _, part in subset.groupby("relevance_degree")]
            labels = [str(v) for v in subset.groupby("relevance_degree").groups]
            if grouped:
                plt.figure(figsize=(7, 5))
                plt.boxplot(grouped, tick_labels=labels)
                plt.xlabel("Ibn Kathir relation degree")
                plt.ylabel("AraBERT semantic similarity")
                plt.title("Similarity by Relation Degree")
                plt.tight_layout()
                plt.savefig(results_dir / "arabert_boxplot.png", dpi=300)
                plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(valid["arabert_similarity"], bins=30)
    plt.xlabel("AraBERT semantic similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of AraBERT Semantic Similarity")
    plt.tight_layout()
    plt.savefig(results_dir / "arabert_histogram.png", dpi=300)
    plt.close()
    print(f"Saved pair analysis charts -> {results_dir}")


# ============================================================================
# 05 | TF-IDF AND PRETRAINED ARABERTv2 EMBEDDINGS
#      (Feature Enginnering.ipynb)
# ============================================================================


def _parse_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a token list in tokens_no_stopwords.")
    return [str(item) for item in parsed]


def make_features(processed_dir: Path, features_dir: Path, model_name: str = RETRIEVAL_MODEL) -> None:
    from sentence_transformers import SentenceTransformer

    quran = get_processed_quran(processed_dir)
    texts_lexical = quran["tokens_no_stopwords"].map(
        lambda v: " ".join(_parse_tokens(v))
    )
    tfidf = TfidfVectorizer()  # original default settings
    tfidf_matrix = tfidf.fit_transform(texts_lexical)
    dump_pickle(tfidf, features_dir / "tfidf_vectorizer.pkl")
    dump_pickle(tfidf_matrix, features_dir / "tfidf_matrix.pkl")

    model = SentenceTransformer(model_name)  # pretrained; not trained here
    texts_semantic = quran["clean_text"].fillna("").astype(str).tolist()
    embeddings = np.asarray(model.encode(texts_semantic, show_progress_bar=True))
    embeddings = np.squeeze(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(quran):
        raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")
    dump_pickle(embeddings, features_dir / "semantic_embeddings.pkl")
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"Semantic embeddings: {embeddings.shape}")
    print(f"Saved features -> {features_dir}")


# ============================================================================
# 06 | HYBRID RETRIEVAL ENGINE
# ============================================================================


class HybridRetrieval:
    def __init__(self, processed_dir: Path, features_dir: Path, model_name: str = RETRIEVAL_MODEL):
        from sentence_transformers import SentenceTransformer

        self.quran = get_processed_quran(processed_dir)
        self.tfidf = load_pickle(features_dir / "tfidf_vectorizer.pkl")
        self.tfidf_matrix = load_pickle(features_dir / "tfidf_matrix.pkl")
        self.embeddings = np.squeeze(np.asarray(load_pickle(features_dir / "semantic_embeddings.pkl")))
        self.model = SentenceTransformer(model_name)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.quran):
            raise ValueError("Corpus and saved embedding rows do not align.")
        if self.tfidf_matrix.shape[0] != len(self.quran):
            raise ValueError("Corpus and saved TF-IDF rows do not align.")

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid",
               alpha: float = DEFAULT_ALPHA, exclude_verse: tuple[int, int] | None = None) -> pd.DataFrame:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if mode not in ("lexical", "semantic", "hybrid"):
            raise ValueError("mode must be lexical, semantic, or hybrid.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        query_clean = preprocess_for_tfidf(query)
        query_tfidf = self.tfidf.transform([query_clean])
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).ravel()
        query_embedding = np.asarray(self.model.encode([query_clean])).reshape(1, -1)
        semantic_scores = cosine_similarity(query_embedding, self.embeddings).ravel()
        if mode == "lexical":
            scores = lexical_scores
        elif mode == "semantic":
            scores = semantic_scores
        else:
            scores = alpha * semantic_scores + (1 - alpha) * lexical_scores

        order = np.argsort(scores)[::-1]
        if exclude_verse is not None:
            s, a = exclude_verse
            mask = (self.quran.iloc[order]["sura"].to_numpy() != s) | (
                self.quran.iloc[order]["ayah"].to_numpy() != a
            )
            order = order[mask]
        best = order[:top_k]
        result = self.quran.iloc[best][["sura", "ayah", "ayah_text"]].copy()
        result["score"] = scores[best]
        return result.reset_index(drop=True)


def get_ground_truth(quran: pd.DataFrame, sura: int, ayah: int) -> str | None:
    found = quran.loc[(quran["sura"] == sura) & (quran["ayah"] == ayah), "ayah_text"]
    return str(found.iloc[0]) if not found.empty else None


# ============================================================================
# 07 | TEST QUERIES AND EVALUATION (notebook's original comparisons)
# ============================================================================

# These expected lists were manually chosen in the supplied notebook, and
# are reproduced here as examples, not as a validated test benchmark.
ORIGINAL_EVALUATION = {
    "lexical": {
        "query": "اهدنا الصراط المستقيم",
        "expected": [(1, 6), (38, 22), (6, 87)],
    },
    "semantic": {
        "query": "إِنَّ فِي ذَٰلِكَ لَآيَاتٍ لِّقَوْمٍ يَتَفَكَّرُونَ",
        "expected": [(45, 13), (2, 164), (2, 230), (30, 21)],
    },
    "hybrid": {
        "query": "من يشفع شفاعة حسنة يكن له نصيب منها",
        "expected": [(4, 85), (2, 48), (2, 123), (2, 254)],
    },
}

HYBRID_ADDITIONAL_CASES = [
    {"query": "من يشفع شفاعة حسنة يكن له نصيب منها",
     "expected": [(4, 85), (2, 48), (2, 123), (2, 254)]},
    {"query": "أَنفِقُوا مِمَّا رَزَقْنَاكُم مِّن قَبْلِ أَن يَأْتِيَ يَوْمٌ لَّا بَيْعٌ فِيهِ وَلَا خُلَّةٌ وَلَا شَفَاعَةٌ",
     "expected": [(2, 254), (2, 48), (2, 123), (4, 85)]},
    {"query": "وَاتَّقُوا يَوْمًا لَّا تَجْزِي نَفْسٌ عَن نَّفْسٍ شَيْئًا وَلَا يُقْبَلُ مِنْهَا عَدْلٌ وَلَا تَنفَعُهَا شَفَاعَةٌ وَلَا هُمْ يُنصَرُونَ",
     "expected": [(2, 123), (2, 48), (2, 254), (4, 85)]},
]


def _relevant_ranks(results: pd.DataFrame, expected: Iterable[tuple[int, int]]) -> list[int]:
    expected_set = set(expected)
    return [rank for rank, (s, a) in enumerate(zip(results["sura"], results["ayah"]), 1)
            if (int(s), int(a)) in expected_set]


def evaluate_one(results: pd.DataFrame, expected: Iterable[tuple[int, int]],
                 k: int = 10, *, legacy_rr: bool = False) -> dict[str, float | int]:
    expected_set = set(expected)
    relevant_ranks = _relevant_ranks(results.head(k), expected_set)
    count = len(relevant_ranks)
    if legacy_rr:
        # Replicate the original notebook's non-standard mean reciprocal rank:
        # average 1/rank across ALL expected verses, with 0 for absent verses.
        rr_value = sum(1.0 / rank for rank in relevant_ranks) / max(len(expected_set), 1)
    else:
        # Standard reciprocal rank: the FIRST relevant result for this query.
        rr_value = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    return {
        "Precision@10": count / k,
        "Recall@10": count / len(expected_set) if expected_set else 0.0,
        "MRR_or_RR": rr_value,
        "Found": count,
        "Expected": len(expected_set),
    }


def evaluate_original(engine: HybridRetrieval, results_dir: Path, k: int = 10,
                      alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Reproduce the original notebook's three DIFFERENT-query scenarios."""
    rows: list[dict[str, Any]] = []
    for mode, sample in ORIGINAL_EVALUATION.items():
        results = engine.search(sample["query"], top_k=k, mode=mode, alpha=alpha)
        scores = evaluate_one(results, sample["expected"], k, legacy_rr=True)
        rows.append({"Model": mode.upper(), "Query": sample["query"], **scores})
        results.to_csv(results_dir / f"original_top10_{mode}.csv", index=False, encoding="utf-8-sig")
    report = pd.DataFrame(rows)
    report.rename(columns={"MRR_or_RR": "Legacy_mean_RR_not_standard_MRR"}, inplace=True)
    out = results_dir / "original_evaluation.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print("ORIGINAL evaluation (different queries; non-standard MRR):")
    print(report.to_string(index=False))
    print(f"Saved -> {out}")
    return report


def evaluate_shared(engine: HybridRetrieval, results_dir: Path, k: int = 10,
                    alpha: float = DEFAULT_ALPHA, exclude_query_verse: bool = False) -> pd.DataFrame:
    """A shared-query illustration, not a fully validated scientific benchmark."""
    rows: list[dict[str, Any]] = []
    for mode in ("lexical", "semantic", "hybrid"):
        for label, sample in ORIGINAL_EVALUATION.items():
            query_id = sample["expected"][0] if exclude_query_verse else None
            expected = sample["expected"][1:] if exclude_query_verse else sample["expected"]
            result = engine.search(sample["query"], top_k=k, mode=mode,
                                   alpha=alpha, exclude_verse=query_id)
            metrics = evaluate_one(result, expected, k, legacy_rr=False)
            rows.append({"Model": mode.upper(), "Query_group": label,
                         "Query": sample["query"], **metrics})
    report = pd.DataFrame(rows)
    report.rename(columns={"MRR_or_RR": "Reciprocal_Rank"}, inplace=True)
    report.to_csv(results_dir / "shared_evaluation_per_query.csv", index=False, encoding="utf-8-sig")
    summary = report.groupby("Model", sort=False)[["Precision@10", "Recall@10", "Reciprocal_Rank"]].mean()
    summary = summary.rename(columns={"Reciprocal_Rank": "MRR"}).reset_index()
    summary.to_csv(results_dir / "shared_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    print("SHARED-QUERY exploratory evaluation (same three queries for every mode):")
    print(summary.to_string(index=False))
    return summary


def evaluate_hybrid_examples(engine: HybridRetrieval, results_dir: Path, k: int = 10,
                             alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    rows = []
    for case in HYBRID_ADDITIONAL_CASES:
        results = engine.search(case["query"], top_k=k, mode="hybrid", alpha=alpha)
        row = evaluate_one(results, case["expected"], k, legacy_rr=True)
        rows.append({"Query": case["query"], **row})
    frame = pd.DataFrame(rows)
    frame.rename(columns={"MRR_or_RR": "Legacy_mean_RR_not_standard_MRR"}, inplace=True)
    frame.to_csv(results_dir / "original_hybrid_examples.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    return frame


def alpha_sweep(engine: HybridRetrieval, results_dir: Path, k: int = 10) -> pd.DataFrame:
    sample = ORIGINAL_EVALUATION["hybrid"]
    rows = []
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
        results = engine.search(sample["query"], top_k=k, mode="hybrid", alpha=alpha)
        rows.append({"Alpha": alpha, **evaluate_one(results, sample["expected"], k, legacy_rr=True)})
    report = pd.DataFrame(rows)
    report.rename(columns={"MRR_or_RR": "Legacy_mean_RR_not_standard_MRR"}, inplace=True)
    report.to_csv(results_dir / "alpha_sweep.csv", index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    return report


# ============================================================================
# 08 | OPTIONAL BM25 BASELINE (Feature Enginnering.ipynb)
# ============================================================================


def bm25_search(quran: pd.DataFrame, query: str, top_k: int = 10) -> pd.DataFrame:
    from rank_bm25 import BM25Okapi
    corpus = quran["clean_text"].fillna("").map(lambda t: str(t).split()).tolist()
    model = BM25Okapi(corpus)
    tokenized_query = preprocess_for_tfidf(query).split()
    scores = model.get_scores(tokenized_query)
    order = np.argsort(scores)[::-1][:top_k]
    result = quran.iloc[order][["sura", "ayah", "ayah_text"]].copy()
    result["score"] = scores[order]
    return result.reset_index(drop=True)


# ============================================================================
# 09 | COMMAND-LINE ENTRY POINT
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-file Quranic preprocessing, AraBERT retrieval and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Place source files under data/; generated artifacts go under output/.",
    )
    parser.add_argument("command", choices=(
        "preprocess", "prepare-pairs", "pair-similarity", "pair-charts",
        "features", "search", "evaluate", "alpha-sweep", "hybrid-examples",
        "bm25", "pipeline",
    ))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query", type=str, default="الحمد لله رب العالمين")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    parser.add_argument("--protocol", choices=("original", "shared"), default="original")
    parser.add_argument("--exclude-query-verse", action="store_true")
    parser.add_argument("--with-pair-analysis", action="store_true",
                        help="Include the optional, slow pair similarity analysis during pipeline.")
    args = parser.parse_args()

    p = dirs(args.data_dir, args.output_dir)
    if args.command in ("preprocess", "pipeline"):
        preprocess_quran(p["data"], p["processed"])
        preprocess_optional_annotations(p["data"], p["processed"])
    if args.command in ("prepare-pairs", "pipeline"):
        if (p["data"] / "ibn_kathir_similarity.csv").exists():
            prepare_verse_pairs(p["data"], p["processed"])
        elif args.command == "prepare-pairs":
            require_file(p["data"] / "ibn_kathir_similarity.csv")
    if args.command == "pair-similarity" or (args.command == "pipeline" and args.with_pair_analysis):
        extract_pair_similarities(p["processed"], p["results"])
    if args.command == "pair-charts" or (args.command == "pipeline" and args.with_pair_analysis):
        pair_analysis_charts(p["processed"], p["results"])
    if args.command in ("features", "pipeline"):
        make_features(p["processed"], p["features"])
    if args.command in ("search", "evaluate", "alpha-sweep", "hybrid-examples", "pipeline"):
        engine = HybridRetrieval(p["processed"], p["features"])
        if args.command == "search":
            print(engine.search(args.query, top_k=args.top_k, mode=args.mode,
                                alpha=args.alpha).to_string(index=False))
        elif args.command == "alpha-sweep":
            alpha_sweep(engine, p["results"], args.top_k)
        elif args.command == "hybrid-examples":
            evaluate_hybrid_examples(engine, p["results"], args.top_k, args.alpha)
        elif args.protocol == "original":
            evaluate_original(engine, p["results"], args.top_k, args.alpha)
        else:
            evaluate_shared(engine, p["results"], args.top_k, args.alpha,
                            exclude_query_verse=args.exclude_query_verse)
    if args.command == "bm25":
        quran = get_processed_quran(p["processed"])
        print(bm25_search(quran, args.query, args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
