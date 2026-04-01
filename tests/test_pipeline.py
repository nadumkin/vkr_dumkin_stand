import tempfile
import unittest
from pathlib import Path

from hybrid_search.core.config import EncoderConfig, HashingModelConfig, SearchConfig, TrainingConfig
from hybrid_search.core.preprocessing import TextPreprocessor
from hybrid_search.core.schemas import SimilarityExample, TextRecord
from hybrid_search.models.encoders import HashingTextEncoder
from hybrid_search.models.hashing import HashingMLP
from hybrid_search.models.training import HashingTrainer, build_triplet_embedding_dataset
from hybrid_search.retrieval.indexing import BinaryCodeIndex
from hybrid_search.retrieval.search import HybridSearchPipeline

try:
    import hnswlib  # noqa: F401
except ImportError:
    HNSWLIB_AVAILABLE = False
else:
    HNSWLIB_AVAILABLE = True


class HybridPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.encoder = HashingTextEncoder(
            EncoderConfig(backend="hashing", embedding_dim=64, batch_size=8, seed=23),
            preprocessor=self.preprocessor,
        )
        self.model = HashingMLP(
            HashingModelConfig(
                input_dim=self.encoder.embedding_dim,
                code_bits=32,
                hidden_dims=(48, 32),
                dropout=0.0,
            )
        )

    def test_training_and_search_end_to_end(self) -> None:
        examples = [
            SimilarityExample(
                query_text="apple fruit nutrition",
                positive_text="fresh apple fruit",
                negative_text="quantum mechanics particle",
            ),
            SimilarityExample(
                query_text="neural semantic retrieval",
                positive_text="semantic search with transformers",
                negative_text="baking chocolate cake",
            ),
        ]
        dataset = build_triplet_embedding_dataset(examples, self.encoder, self.preprocessor)
        trainer = HashingTrainer(
            self.model,
            TrainingConfig(epochs=2, batch_size=2, learning_rate=5e-3, device="cpu"),
        )
        history = trainer.fit(dataset)
        self.assertEqual(len(history), 2)

        records = [
            TextRecord(record_id="doc-apple", text="fresh apple fruit and vitamins"),
            TextRecord(record_id="doc-search", text="semantic search with transformer embeddings"),
            TextRecord(record_id="doc-physics", text="quantum mechanics and particle spin"),
        ]
        pipeline = HybridSearchPipeline(self.preprocessor, self.encoder, self.model, device="cpu")
        pipeline.index_documents(records)

        apple_results = pipeline.search(
            "apple fruit",
            config=SearchConfig(strategy="coarse_rerank", top_k=2, oversample_factor=3),
        )
        self.assertEqual(apple_results[0].record_id, "doc-apple")

        search_results = pipeline.search(
            "semantic transformer retrieval",
            config=SearchConfig(strategy="integrated_filtering", top_k=2, oversample_factor=3),
        )
        self.assertEqual(search_results[0].record_id, "doc-search")

    def test_index_roundtrip(self) -> None:
        records = [
            TextRecord(record_id="a", text="alpha"),
            TextRecord(record_id="b", text="beta"),
        ]
        pipeline = HybridSearchPipeline(self.preprocessor, self.encoder, self.model, device="cpu")
        pipeline.index_documents(records)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.index.save(tmp_dir)
            loaded = BinaryCodeIndex.load(tmp_dir)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.records[0].record_id, "a")

    @unittest.skipUnless(HNSWLIB_AVAILABLE, "hnswlib is not available in the current interpreter")
    def test_hnsw_backend_search(self) -> None:
        records = [
            TextRecord(record_id="doc-apple", text="fresh apple fruit and vitamins"),
            TextRecord(record_id="doc-search", text="semantic search with transformer embeddings"),
            TextRecord(record_id="doc-physics", text="quantum mechanics and particle spin"),
        ]
        index = BinaryCodeIndex(code_bits=32, backend="hnswlib", hnsw_m=8, hnsw_ef_construction=50, hnsw_ef_search=20)
        pipeline = HybridSearchPipeline(self.preprocessor, self.encoder, self.model, index=index, device="cpu")
        pipeline.index_documents(records)
        self.assertEqual(pipeline.index.backend, "hnswlib")

        results = pipeline.search(
            "semantic transformer retrieval",
            config=SearchConfig(strategy="coarse_rerank", top_k=2, oversample_factor=2),
        )
        self.assertEqual(results[0].record_id, "doc-search")


if __name__ == "__main__":
    unittest.main()
