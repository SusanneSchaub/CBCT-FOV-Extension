import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own
import cbct_artifact_reduction.pigjawdataset as dataset
from cbct_artifact_reduction.pigjawdataset_nolfs import PairedNiftiDataset
from cbct_artifact_reduction.argparser_config import create_train_argparser
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.resample import (
    create_named_schedule_sampler,
)
from cbct_artifact_reduction.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from cbct_artifact_reduction.guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
import numpy as np
import random
import torch as th
SEED = 42

# Python
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)



def main():
    args = create_train_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(os.path.expanduser(args.log_dir))

    today = datetime.now()
    logger.log(f"TRAINING {today}")
    logger.log(f"args: {args}")
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    #model.to("cuda")
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
    )

    dataset = PairedNiftiDataset(
        root_dir=""
    )


    logger.log("creating dataloader...")


    num_epochs = args.num_epochs
    logger.log("training...")
    step = 0
    for epoch in range(num_epochs):
        logger.log(f"epoch {epoch + 1}/{num_epochs}")
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers= args.batch_size,
        )
        data = iter(dataloader)

        step = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            step=step if step else 0,
        ).run_loop()
        # Make sure to not resume checkpoint again after first epoch:
        args.resume_checkpoint = ""


if __name__ == "__main__":
    main()
