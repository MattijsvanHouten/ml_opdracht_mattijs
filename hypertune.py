from src.data import make_dataset
from src.models import rnn_models, metrics, train_model
from src.datasets import get_arabic
from src.settings import SearchSpace, presets
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from typing import Dict
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from loguru import logger
from filelock import FileLock
import urllib.parse as up


def train(config: Dict, checkpoint_dir=None):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """

    # we lock the datadir to avoid parallel instances trying to
    # access the datadir
    # data_dir = config["data_dir"]
    # with FileLock(data_dir / ".lock"):
    #     trainloader, testloader = make_dataset.get_gestures(
    #         data_dir=data_dir, split=0.8, batchsize=32
    #     )

    trainstreamer, teststreamer = get_arabic(presets)

    # we set up the metric
    accuracy = metrics.Accuracy()
    # and create the model with the config
    model = rnn_models.GRUmodel(config)

    # and we start training.
    # because we set tunewriter=True
    # the trainloop wont try to report back to tensorboard,
    # but will report back with tune.report
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop
    model = train_model.trainloop(
        epochs=50,
        model=model,
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy],
        train_dataloader=trainstreamer.stream(),
        test_dataloader=teststreamer.stream(),
        log_dir=".",
        train_steps=len(trainstreamer),
        eval_steps=len(teststreamer),
        patience=5,
        factor=0.5,
        tunewriter=["ray"],
    )


if __name__ == "__main__":
    ray.init()

    # have a look in src.settings to see how SearchSpace is created.
    # If you want to search other ranges, you change this in the settings file.
    config = SearchSpace(
        input_size=13,
        output_size=20,
        tune_dir=Path("models/ray").resolve(),
        data_dir=Path("data/raw").resolve(),
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config.dict(),
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=config.tune_dir,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()