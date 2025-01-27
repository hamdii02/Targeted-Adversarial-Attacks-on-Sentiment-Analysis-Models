from pytest import approx
from chameleon.models import HuggingFaceModel, BaseModel


def test_can_load_huggingface_model():
    model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    assert isinstance(model, BaseModel)
    assert isinstance(model, HuggingFaceModel)

    assert set(model.labels()) == set(["negative", "neutral", "positive"])

    pred = model.predict("Hello world!")
    assert pred["positive"] == approx(0.783, abs=1e-3)
    assert pred["neutral"] == approx(0.1, abs=1e-3)
