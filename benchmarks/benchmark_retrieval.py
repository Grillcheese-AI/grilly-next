"""
CubeMind VSA Retrieval Benchmark on MS MARCO
=============================================

End-to-end test of the grilly C++ pipeline:
  MiniLM sentence embeddings (ONNX) -> LSH random projection -> bipolar ->
  VSACache (Vulkan Hamming distance search)

Measures:
  - Hit Rate @ 5 / @ 10
  - MRR (Mean Reciprocal Rank)
  - NDCG @ 10
  - GPU query latency (wall-clock + shader time)
  - Indexing throughput (docs/sec)

Uses MS MARCO v1.1 from HuggingFace datasets.
Sentence embeddings via all-MiniLM-L6-v2 (384D) through ONNX Runtime.
LSH Gaussian projection to 10240D bipolar for Hamming search on GPU.
"""

import math
import os
import time

import numpy as np


# == MS MARCO loader ======================================================


def load_msmarco(num_queries=500, streaming=True):
    """Load MS MARCO v1.1 and extract query->passage retrieval pairs.

    Returns:
        corpus: list of {"id": str, "text": str}
        queries: list of {"text": str, "relevant_doc_ids": list[str]}
    """
    from datasets import load_dataset

    print("Loading MS MARCO v1.1 (streaming)...")
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=streaming)

    corpus = []
    queries = []
    doc_id_counter = 0
    passage_to_id = {}  # dedup: passage text hash -> doc_id

    for i, sample in enumerate(ds):
        if i >= num_queries:
            break

        query_text = sample["query"]
        passages = sample["passages"]["passage_text"]
        is_selected = sample["passages"]["is_selected"]

        relevant_ids = []

        for passage, selected in zip(passages, is_selected):
            key = passage[:200]
            if key in passage_to_id:
                doc_id = passage_to_id[key]
            else:
                doc_id = f"d{doc_id_counter}"
                doc_id_counter += 1
                passage_to_id[key] = doc_id
                corpus.append({"id": doc_id, "text": passage})

            if selected:
                relevant_ids.append(doc_id)

        if relevant_ids:
            queries.append({
                "text": query_text,
                "relevant_doc_ids": relevant_ids,
            })

    print(f"  Loaded {len(corpus)} passages, {len(queries)} queries "
          f"with relevance labels")
    return corpus, queries


# == Sentence Encoder (ONNX MiniLM + LSH projection) ======================


class SentenceEncoder:
    """Encode sentences to bipolar VSA vectors via:
    1. all-MiniLM-L6-v2 (384D dense embeddings via ONNX Runtime)
    2. LSH Gaussian random projection (384D -> dim bipolar {-1,+1})

    The Johnson-Lindenstrauss lemma guarantees that cosine similarity
    in the 384D MiniLM space is preserved as Hamming distance in the
    high-dimensional bipolar space.
    """

    def __init__(self, dim=10240, batch_size=64):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.dim = dim
        self.batch_size = batch_size
        self.embed_dim = 384  # MiniLM-L6-v2 output dimension

        # Load tokenizer
        print("  Loading MiniLM tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Download + load ONNX model
        print("  Loading MiniLM ONNX model...")
        model_path = hf_hub_download(
            "sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx"
        )
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Build deterministic LSH projection matrix: R^384 -> R^dim
        # Seed 42 ensures reproducibility across sessions
        print(f"  Building LSH projection matrix ({self.embed_dim} -> {dim})...")
        rng = np.random.RandomState(42)
        self.projection = rng.randn(self.embed_dim, dim).astype(np.float32)
        # Normalize columns for unit-variance projections
        col_norms = np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.projection /= col_norms

    def encode_dense(self, texts):
        """Encode texts to 384D dense normalized embeddings (batched)."""
        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=128,
            )
            outputs = self.session.run(None, dict(encoded))
            token_emb = outputs[0]  # (B, seq_len, 384)
            attn_mask = encoded["attention_mask"]

            # Mean pooling
            mask_exp = np.expand_dims(attn_mask, -1).astype(np.float32)
            pooled = (token_emb * mask_exp).sum(axis=1) / np.maximum(
                mask_exp.sum(axis=1), 1e-9
            )

            # L2 normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norms, 1e-9)

            all_embeddings.append(pooled)

        return np.vstack(all_embeddings).astype(np.float32)

    def project_to_bipolar(self, dense_embeddings):
        """LSH project dense embeddings to bipolar {-1, +1} int8.

        bipolar[i,j] = sign( dense[i,:] @ projection[:,j] )
        """
        projected = dense_embeddings @ self.projection  # (N, dim)
        bipolar = np.sign(projected).astype(np.int8)
        # sign(0) = 0, snap to +1
        bipolar[bipolar == 0] = 1
        return bipolar

    def encode(self, texts):
        """Full pipeline: texts -> dense -> bipolar int8 vectors."""
        dense = self.encode_dense(texts)
        return self.project_to_bipolar(dense)


# == Metrics ===============================================================


def hit_at_k(retrieved, relevant, k):
    return int(any(doc_id in relevant for doc_id in retrieved[:k]))


def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


# == Benchmark =============================================================


def run_benchmark(num_queries=500, dim=10240, top_k=10):
    """Run the full retrieval benchmark.

    Pipeline:
      1. Load MS MARCO (streaming, first `num_queries` samples)
      2. all-MiniLM-L6-v2 encodes all passages + queries to 384D
      3. LSH Gaussian projection to dim-D bipolar {-1,+1}
      4. VSACache: Vulkan GPU Hamming distance retrieval
      5. Compute Hit@5, Hit@10, MRR, NDCG@10, latency stats
    """
    import grilly_core

    # -- Load data ---------------------------------------------------------
    corpus, queries = load_msmarco(num_queries=num_queries)

    if not queries:
        print("ERROR: No queries with relevance labels found.")
        return

    # -- Initialize engines ------------------------------------------------
    print("\nInitializing engines...")
    dev = grilly_core.Device()
    dev.load_shaders("shaders/spv")
    print(f"  GPU: {dev.device_name}")

    encoder = SentenceEncoder(dim=dim, batch_size=64)

    # -- Encode corpus -----------------------------------------------------
    corpus_texts = [doc["text"] for doc in corpus]
    print(f"\nEncoding {len(corpus_texts)} passages with MiniLM + LSH...")
    t_enc_start = time.perf_counter()
    corpus_bipolar = encoder.encode(corpus_texts)
    t_enc_elapsed = time.perf_counter() - t_enc_start
    print(f"  Encoded in {t_enc_elapsed:.2f}s "
          f"({len(corpus_texts) / t_enc_elapsed:.0f} docs/sec)")
    print(f"  Bipolar shape: {corpus_bipolar.shape}, "
          f"dtype: {corpus_bipolar.dtype}")

    # -- Encode queries ----------------------------------------------------
    query_texts = [q["text"] for q in queries]
    print(f"\nEncoding {len(query_texts)} queries...")
    t_qenc_start = time.perf_counter()
    query_bipolar = encoder.encode(query_texts)
    t_qenc_elapsed = time.perf_counter() - t_qenc_start
    print(f"  Encoded in {t_qenc_elapsed:.2f}s")

    # -- Build VSACache ----------------------------------------------------
    print(f"\nLoading {len(corpus)} vectors into VSACache...")
    cache = grilly_core.VSACache(
        dev,
        initial_capacity=min(len(corpus), 2048),
        max_capacity=max(len(corpus) + 1000, 10000),
        dim=dim,
        surprise_threshold=0.0,
        utility_decay=0.99,
    )

    t_insert_start = time.perf_counter()
    for vec in corpus_bipolar:
        cache.insert(vec, surprise=1.0, stress=0.0)
    t_insert_elapsed = time.perf_counter() - t_insert_start
    print(f"  Inserted {cache.size()} entries in {t_insert_elapsed:.3f}s "
          f"({cache.size() / t_insert_elapsed:.0f} inserts/sec)")

    # -- Build HammingSearchBench for raw GPU timing -----------------------
    hsb = grilly_core.HammingSearchBench(dev, corpus_bipolar, dim=dim)
    print(f"  HammingSearchBench: {corpus_bipolar.shape[0]} entries in VRAM")

    # -- Run retrieval -----------------------------------------------------
    print(f"\nRunning retrieval (top-{top_k}) on {len(queries)} queries...")
    doc_ids = [doc["id"] for doc in corpus]

    hits_5 = 0
    hits_10 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    latencies_wall = []
    latencies_gpu = []

    for qi, (q, qvec) in enumerate(zip(queries, query_bipolar)):
        relevant = set(q["relevant_doc_ids"])

        # VSACache lookup (Vulkan Hamming search)
        t0 = time.perf_counter()
        result = cache.lookup(dev, qvec, top_k=top_k)
        t1 = time.perf_counter()
        latencies_wall.append((t1 - t0) * 1000)

        # Pure GPU shader time via HammingSearchBench
        _ = hsb.search_top1(qvec)
        latencies_gpu.append(hsb.gpu_time_ms)

        # Map indices to doc IDs
        indices = result["indices"]
        retrieved = [doc_ids[idx] for idx in indices if idx < len(doc_ids)]

        # Metrics
        hits_5 += hit_at_k(retrieved, relevant, 5)
        hits_10 += hit_at_k(retrieved, relevant, 10)
        mrr_sum += reciprocal_rank(retrieved, relevant)
        ndcg_sum += ndcg_at_k(retrieved, relevant, top_k)

    # -- Results -----------------------------------------------------------
    n = len(queries)
    print("\n" + "=" * 60)
    print("  CubeMind VSA Retrieval Benchmark -- MS MARCO v1.1")
    print("=" * 60)
    print(f"  Corpus:          {len(corpus)} passages")
    print(f"  Queries:         {n} with relevance labels")
    print(f"  VSA dimension:   {dim}")
    print(f"  Embedding:       all-MiniLM-L6-v2 (384D) via ONNX")
    print(f"  Projection:      LSH Gaussian (384 -> {dim})")
    print(f"  GPU:             {dev.device_name}")
    print()
    print("  -- Retrieval Quality --")
    print(f"  Hit Rate @ 5:    {hits_5 / n * 100:.1f}%")
    print(f"  Hit Rate @ 10:   {hits_10 / n * 100:.1f}%")
    print(f"  MRR:             {mrr_sum / n:.4f}")
    print(f"  NDCG @ 10:       {ndcg_sum / n:.4f}")
    print()
    print("  -- Latency (VSACache.lookup) --")
    print(f"  Mean:            {np.mean(latencies_wall):.3f} ms")
    print(f"  Median:          {np.median(latencies_wall):.3f} ms")
    print(f"  P95:             {np.percentile(latencies_wall, 95):.3f} ms")
    print(f"  P99:             {np.percentile(latencies_wall, 99):.3f} ms")
    print()
    print("  -- GPU Shader Time (HammingSearchBench) --")
    print(f"  Mean:            {np.mean(latencies_gpu):.4f} ms")
    print(f"  P99:             {np.percentile(latencies_gpu, 99):.4f} ms")
    print()
    print("  -- Throughput --")
    print(f"  Encoding:        {len(corpus) / t_enc_elapsed:.0f} docs/sec")
    print(f"  Indexing:        {cache.size() / t_insert_elapsed:.0f} inserts/sec")
    qps = n / sum(l / 1000 for l in latencies_wall) if latencies_wall else 0
    print(f"  Query:           {qps:.0f} queries/sec")
    print("=" * 60)

    return {
        "hit_at_5": hits_5 / n,
        "hit_at_10": hits_10 / n,
        "mrr": mrr_sum / n,
        "ndcg_at_10": ndcg_sum / n,
        "mean_latency_ms": np.mean(latencies_wall),
        "p99_latency_ms": np.percentile(latencies_wall, 99),
        "gpu_shader_ms": np.mean(latencies_gpu),
        "corpus_size": len(corpus),
        "num_queries": n,
    }


if __name__ == "__main__":
    run_benchmark(num_queries=500, dim=10240, top_k=10)
