from . import nnfabrik
from nnfabrik.templates.checkpoint import TrainedModelChkptBase, my_checkpoint

Checkpoint = my_checkpoint(nnfabrik)

@nnfabrik.schema
class TrainedModel(TrainedModelChkptBase):
    table_comment = "My Trained models"
    nnfabrik = nnfabrik
    checkpoint_table = Checkpoint
