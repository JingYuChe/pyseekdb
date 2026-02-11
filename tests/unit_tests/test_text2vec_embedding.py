"""
Unit tests for Text2VecEmbeddingFunction
"""

import sys
from unittest.mock import MagicMock

# Pre-mock text2vec to avoid import errors (like missing _lzma) in restricted environments
mock_text2vec = MagicMock()
mock_sentence_model = MagicMock()
mock_text2vec.SentenceModel = mock_sentence_model
sys.modules["text2vec"] = mock_text2vec

from pyseekdb.utils.embedding_functions import Text2VecEmbeddingFunction  # noqa: E402


class TestText2VecEmbeddingFunction:
    """Test Text2VecEmbeddingFunction class"""

    def setup_method(self):
        # Reset mocks before each test
        mock_text2vec.reset_mock()
        mock_sentence_model.reset_mock()
        Text2VecEmbeddingFunction.models.clear()

    def test_init_defaults(self):
        """Test initialization with default parameters"""
        _ef = Text2VecEmbeddingFunction()
        assert _ef.model_name == "shibing624/text2vec-base-chinese"
        assert _ef.device == "cpu"
        assert _ef.normalize_embeddings is False

        # Verify model was NOT initialized yet (lazy loading)
        mock_sentence_model.assert_not_called()

        # Trigger model loading
        _ef._get_model()

        # Verify model was initialized
        mock_sentence_model.assert_called_with(model_name_or_path="shibing624/text2vec-base-chinese", device="cpu")

    def test_call(self):
        """Test embedding generation"""
        # Setup mock instance behavior
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_model.return_value = mock_instance

        # Initialize
        ef = Text2VecEmbeddingFunction(model_name="test-model")

        # Test call
        documents = ["test document"]
        embeddings = ef(documents)

        # Verification
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
        assert embeddings == [[0.1, 0.2, 0.3]]

        # Verify mock was called correctly
        mock_instance.encode.assert_called_with(["test document"], normalize_embeddings=False)

    def test_dimension(self):
        """Test dimension property"""
        mock_instance = MagicMock()
        # Mock encode for dimension check
        # Simulate numpy array shape behavior
        mock_array = MagicMock()
        mock_array.shape = (1, 3)
        mock_array.__len__.return_value = 1
        mock_instance.encode.return_value = mock_array

        mock_sentence_model.return_value = mock_instance

        ef = Text2VecEmbeddingFunction(model_name="test-model")
        assert ef.dimension == 3

    def test_config_serialization(self):
        """Test get_config and build_from_config"""
        config = {
            "model_name": "custom-model",
            "device": "cuda",
            "normalize_embeddings": True,
            "kwargs": {"cache_folder": "test_cache"},
        }

        ef = Text2VecEmbeddingFunction.build_from_config(config)

        assert ef.model_name == "custom-model"
        assert ef.device == "cuda"
        assert ef.normalize_embeddings is True
        assert ef.kwargs == {"cache_folder": "test_cache"}

        # Test get_config
        retrieved_config = ef.get_config()
        assert retrieved_config == config

    def test_lazy_loading(self):
        """Test that model is loaded only when needed"""
        # Clear cache first
        Text2VecEmbeddingFunction.models.clear()
        mock_sentence_model.reset_mock()

        # First init should NOT load model immediately
        _ef = Text2VecEmbeddingFunction(model_name="new-model")
        mock_sentence_model.assert_not_called()

        # First usage should load model
        _ef._get_model()
        mock_sentence_model.assert_called_once()

        # Second usage with same config should not call constructor again
        _ef._get_model()
        mock_sentence_model.assert_called_once()

        # Another instance with same config should reuse cached model
        _ef2 = Text2VecEmbeddingFunction(model_name="new-model")
        _ef2._get_model()
        mock_sentence_model.assert_called_once()
