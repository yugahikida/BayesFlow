import copy
import keras
import pytest


from tests.utils import assert_models_equal
from tests.utils import InterruptFitCallback, FitInterruptedError


@pytest.mark.parametrize("jit_compile", [False, True])
def test_compile(approximator, random_samples, jit_compile):
    approximator.compile(jit_compile=jit_compile)


@pytest.mark.flaky(reruns=1, only_rerun="AssertionError")
def test_fit(approximator, train_dataset, validation_dataset, batch_size):
    from bayesflow.metrics import MaximumMeanDiscrepancy

    approximator.compile(inference_metrics=[MaximumMeanDiscrepancy()])

    mock_data = train_dataset[0]
    mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
    approximator.build_from_data(mock_data)

    untrained_weights = copy.deepcopy(approximator.weights)
    untrained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    approximator.fit(dataset=train_dataset, epochs=20, batch_size=batch_size)

    trained_weights = approximator.weights
    trained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    # check weights have changed during training
    assert any([keras.ops.any(~keras.ops.isclose(u, t)) for u, t in zip(untrained_weights, trained_weights)])

    assert isinstance(untrained_metrics, dict)
    assert isinstance(trained_metrics, dict)

    # test that metrics are improving
    for metric in ["loss", "maximum_mean_discrepancy/inference_maximum_mean_discrepancy"]:
        assert metric in untrained_metrics
        assert metric in trained_metrics
        assert trained_metrics[metric] <= untrained_metrics[metric]


@pytest.mark.parametrize("jit_compile", [False, True])
def test_serialize_deserialize(tmp_path, approximator, train_dataset, jit_compile):
    mock_data = train_dataset[0]
    mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
    approximator.build_from_data(mock_data)

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded_approximator = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded_approximator)


@pytest.mark.skip(reason="not implemented")
def test_interrupt_and_resume_fit(tmp_path, amortizer, dataset):
    # TODO: test the InterruptFitCallback
    amortizer.compile(optimizer="AdamW")

    callbacks = [
        InterruptFitCallback(epochs=1, error_type=FitInterruptedError),
        keras.callbacks.ModelCheckpoint(tmp_path / "model.keras"),
    ]

    with pytest.raises(FitInterruptedError):
        # interrupted fit (epochs 0-1)
        amortizer.fit(dataset, epochs=2, callbacks=callbacks)

    assert (tmp_path / "model.keras").exists(), "checkpoint has not been created"

    loaded_amortizer = keras.saving.load_model(tmp_path / "model.keras")

    # TODO: verify the fit is actually resumed (and not just started new with the existing weights)
    #  might require test code change
    # resume fit (epochs 1-2)
    loaded_amortizer.fit(dataset, epochs=2)
