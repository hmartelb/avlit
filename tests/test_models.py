import pytest

from src.avlit import AVLIT
from tests.utils import model_forward_test


class TestModels:
    """
    Class to test all the models, where each model is defined in an individual function.
    Model forward test ``_model_forward_test()`` is called after instantiating the model given the expected input and output shapes.
    """

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("num_sources", [1, 2])
    @pytest.mark.parametrize("sr", [8000, 16000])
    @pytest.mark.parametrize("segment_length", [4])
    @pytest.mark.parametrize("fps", [25])
    @pytest.mark.parametrize("audio_num_blocks", [2, 4, 8])
    @pytest.mark.parametrize("video_num_blocks", [1, 2, 4])
    @pytest.mark.parametrize("video_embedding_dim", [1024])
    @pytest.mark.parametrize("fusion_operation", ["sum", "prod", "concat"])
    def test_avlit(
        self,
        batch_size,
        num_sources,
        sr,
        segment_length,
        fps,
        audio_num_blocks,
        video_num_blocks,
        video_embedding_dim,
        fusion_operation,
    ):
        # Instantiate the model
        model = AVLIT(
            num_sources=num_sources,
            audio_num_blocks=audio_num_blocks,
            video_num_blocks=video_num_blocks,
            video_embedding_dim=video_embedding_dim,
            fusion_operation=fusion_operation,
            fusion_positions=[0],
        )

        # Generate expected I/O shapes
        input_shape = [
            (batch_size, 1, segment_length * sr),  # Audio mixture
            (
                batch_size,
                num_sources,
                fps * segment_length,
                64,
                64,
            ),  # Video inputs (1 video per speaker)
        ]
        output_shape = [
            (batch_size, num_sources, segment_length * sr),
        ]

        # Test the model
        model_forward_test(model, input_shape, output_shape, strict=False)
