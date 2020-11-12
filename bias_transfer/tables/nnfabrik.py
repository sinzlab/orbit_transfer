import datajoint as dj

# importing the tables here is a trick to get IntelliSense to work
from nnfabrik.main import Fabrikant, Trainer, Dataset, Model, Seed, my_nnfabrik
from nnfabrik.templates.transfer.recipes import TrainedModelTransferRecipe

# define nnfabrik tables here
my_nnfabrik(
    dj.config["custom"].get("nnfabrik.my_schema_name"),
    additional_tables=(TrainedModelTransferRecipe,),
    context=locals(),
)
