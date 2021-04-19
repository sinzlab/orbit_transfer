from . import ClassificationModel


class ToyModel(ClassificationModel):
    fn = "bias_transfer.models.classification_model_builder"

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.type: str = "linear"
        self.num_classes: int = 1
        self.input_size: int = 2
        super().__init__(**kwargs)
