from pathlib import Path
from typing import cast, Dict, Optional, Union

from pydantic import BaseModel, HttpUrl, root_validator
from ray import tune

import urllib.parse as up

cwd = Path(__file__)
root = (cwd / "../..").resolve()

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

class BaseSearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values


# this is what ray will use to create configs
class SearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 5)


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


# note pydantic handles perfectly a string as url
# but mypy doesnt know that, so to keep mypy satisfied
# i am adding the "cast" for the urls.
presets = Settings(
    datadir=root / "data/raw",
    testurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    ),
    trainurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    ),
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=64,
)