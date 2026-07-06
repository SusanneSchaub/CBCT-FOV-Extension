import argparse

import cbct_artifact_reduction.guided_diffusion.script_util as script_util


def create_train_argparser():
    defaults = dict(
        data_dir="",
        model_dir="models",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        random_masks=True,
        num_epochs=10000,
        num_channels=128,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions=16,
        diffusion_steps=2, #1000, #change to 2
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        lakefs_folder="",
        data_csv="training_data.csv",
        augment_data=True,
        log_transform=True,
        beta_min=0.1,
        beta_max=20,
    )
    defaults.update(script_util.model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_util.add_dict_to_argparser(parser, defaults)
    return parser


def create_sample_argparser():
    defaults = dict(
        log_dir="samples",
        model_dir="",
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path1="",
        model_path2="",
        image_size=256,
        random_masks=True,
        data_csv="sample_data.csv",
        lakefs_folder="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_epochs=10000,
        num_channels=128,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions=16,
        diffusion_steps=2, #1000, #change to 2
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        augment_data=False,
        log_transform=True,
        beta_min=0.1,
        beta_max=20,
    )
    defaults.update(script_util.model_and_diffusion_defaults())  # type: ignore
    parser = argparse.ArgumentParser()
    script_util.add_dict_to_argparser(parser, defaults)
    return parser
