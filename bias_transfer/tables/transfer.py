from . import nnfabrik
from nnfabrik.templates.transfer.checkpoint import TransferredTrainedModelChkptBase, my_checkpoint

Checkpoint = my_checkpoint(nnfabrik)

@nnfabrik.schema
class TransferredTrainedModel(TransferredTrainedModelChkptBase):
    table_comment = "Transferred trained models"
    nnfabrik = nnfabrik
    checkpoint_table = Checkpoint
