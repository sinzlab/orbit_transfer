from nnfabrik.main import *
from nnfabrik.templates.trained_model_chkpts import TrainedModelChkptBase
from .collapse import Collapsed


@schema
class TrainedModel(TrainedModelChkptBase):
    table_comment = "My Trained models"


@schema
class CollapsedTrainedModel(Collapsed):
    Source = TrainedModel()
