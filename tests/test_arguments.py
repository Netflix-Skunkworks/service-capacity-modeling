from service_capacity_modeling.models.org.netflix import models


def test_model_arguments():
    for model in models().values():
        schema = model.extra_model_arguments_schema()
        assert schema is not None
