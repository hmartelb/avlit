import torch


def shapes_equal(x: torch.Tensor, y: torch.Tensor):
    assert (
        x.shape == y.shape
    ), f"Shapes do not match, expected {x.shape} but got {y.shape}"


def data_equal(x: torch.Tensor, y: torch.Tensor, eps=1e-6):
    assert torch.allclose(
        x, y, atol=eps
    ), f"Inputs are not equal, mean absolute difference = {(x-y).abs().sum() / torch.numel(x)}"


def model_forward_test(
    model,
    input_shape,
    output_shape,
    strict=True,
):
    """
    Generic method to test the forward function of a given model.
    The expected input and output shape(s) should be provided.

    Args:
        - model (torch.nn.Module): Model to be tested, must implement ``forward()`` or ``__call__()`` methods.
        - input_shape (Tuple, list): Shape of the input tensor.
        - output_shape (Tuple, list): Shape of the output tensor.
        - strict (bool, default: True): Check that the results of forward for ``train()`` and ``eval()`` are consistent. Must be set to ``False`` for the tests to pass if the model contains instances of ``nn.Dropout`` or ``nn.BatchNorm`` since they have different behaviors for training and inference.
    """
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]

    if isinstance(output_shape, tuple):
        output_shape = [output_shape]

    # Generate some random input data
    x = [torch.randn(s) for s in input_shape]

    last_output = None
    for mode in ["train", "eval"]:
        # Set the model to training or inference mode
        model = model.train() if mode == "train" else model.eval()

        # Forward operation
        y = model(*x)

        if isinstance(y, torch.Tensor):
            y = [y]

        # Tests:
        # 1) Check that the output shape is the one expected
        for i in range(len(output_shape)):
            assert (
                y[i].shape == output_shape[i]
            ), f"[{mode.upper()}] Output shape does not match. Expected {output_shape[i]}, got {y[i].shape}."

        # 2) Check that the outputs are consistent for the same input in train() and eval()
        if last_output is not None and strict:
            for i in range(len(output_shape)):
                assert data_equal(
                    y[i], last_output[i]
                ), "Output mismatch between TRAIN and EVAL"

        # Save the last output
        last_output = y
