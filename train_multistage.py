import numpy as np
from slar.io import PhotonLibDataset, PLibDataLoader
from slar.siren_vis import SirenVis
import torch
import yaml
from slar.io import PLibDataLoader
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device
import loss as custom_loss
from sirent_vis import SirenTVis
import os
import time
from tqdm import tqdm


def train(cfg: dict):
    """
    A function to run an optimization loop for SirenVis model.
    Configuration specific to this function is "train" at the top level.

    Parameters
    ----------
    max_epochs : int
        The maximum number of epochs before stopping training

    max_iterations : int
        The maximum number of iterations before stopping training

    save_every_epochs : int
        A period in epochs to store the network state

    save_every_iterations : int
        A period in iterations to store the network state

    optimizer_class : str
        An optimizer class name to train SirenVis

    optimizer_param : dict
        Optimizer constructor arguments

    resume : bool
        If True, and if a checkopint file is provided for the model, resume training
        with the optimizer state restored from the last checkpoint step.

    """

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.get("device"):
        DEVICE = get_device(cfg["device"]["type"])

    iteration_ctr = 0
    epoch_ctr = 0

    # Create necessary pieces: the model, optimizer, loss, logger.
    # Load the states if this is resuming.
    net = SirenTVis(cfg).to(DEVICE)
    print(net)
    dl = PLibDataLoader(cfg, device=DEVICE)

    opt, sch, epoch = optimizer_factory(net.parameters(), cfg)
    if epoch > 0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        print(
            "[train] resuming training from iteration",
            iteration_ctr,
            "epoch",
            epoch_ctr,
        )

    num_outputs = net.n_outs
    loss_fns = cfg.get("train", dict()).get(
        "loss_fn", {}).get("functions", ["WeightedL2Loss" for _ in range(num_outputs)])
    loss_fns = [getattr(custom_loss, loss_fn)() for loss_fn in loss_fns]
    loss_fn_weights = cfg.get("train", dict()).get(
        "loss_fn", {}).get("weights", [1.0 for _ in range(num_outputs)])

    logger = CSVLogger(cfg)
    logdir = logger.logdir

    # Set the control parameters for the training loop
    train_cfg = cfg.get("train", dict())
    epoch_max = train_cfg.get("max_epochs", int(1e20))
    iteration_max = train_cfg.get("max_iterations", int(1e20))
    save_every_iterations = train_cfg.get("save_every_iterations", -1)
    save_every_epochs = train_cfg.get("save_every_epochs", -1)
    print(f"[train] train for max iterations {iteration_max} or max epochs {epoch_max}")

    # Store configuration
    with open(os.path.join(logdir, "train_cfg.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # Start the training loop
    t0 = time.time()
    twait = time.time()
    stop_training = False
    losses = [np.inf, np.inf]
    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        for batch_idx, data in enumerate(tqdm(dl, desc="Epoch %-3d; Loss %-3s" % (epoch_ctr, ",".join(["%.2e" % loss for loss in losses])))):
            iteration_ctr += 1

            # Input data prep
            x = data["position"].contiguous()  # .to(DEVICE)
            weights = data["weight"].contiguous()  # .to(DEVICE)
            target = data["target"].contiguous()  # .to(DEVICE)
            target_linear = data["value"].contiguous()  # .to(DEVICE)

            twait = time.time() - twait
            # Running the model, compute the loss, back-prop gradients to optimize.
            ttrain = time.time()
            pred = net(x)


            loss = 0
            losses = []
            feature_ctr = 0
            for idx, features in enumerate(net.out_features):
                curr_loss = loss_fn_weights[idx] * loss_fns[idx](
                    pred[:, feature_ctr : feature_ctr + features],
                    target[:, feature_ctr : feature_ctr + features],
                    weights[:, feature_ctr : feature_ctr + features],
                )
                loss += curr_loss
                losses.append(curr_loss.item())

                feature_ctr += features
            loss /= len(net.out_features)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ttrain = time.time() - ttrain

            # Log training parameters
            logger.record(
                ["iter", "epoch", "ttrain", "twait"] + [f'loss_{i}' for i in range(len(losses))],
                [iteration_ctr, epoch_ctr, ttrain, twait] + losses,
            )

            # Step the logger
            pred_linear = dl.inv_xform_vis(pred)
            logger.step(iteration_ctr, target_linear, pred_linear)
            twait = time.time()

            # Save the model parameters if the condition is met
            if save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(
                    logdir,
                    "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr),
                )
                net.save_state(filename, opt, sch, iteration_ctr)

            if iteration_max <= iteration_ctr:
                stop_training = True
                break

        if stop_training:
            break

        if sch is not None:
            sch.step()

        epoch_ctr += 1

        if (save_every_epochs * epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(
                logdir, "iteration-%06d-epoch-%04d.ckpt" % (iteration_ctr, epoch_ctr)
            )
            net.save_state(filename, opt, sch, iteration_ctr / len(dl))

    print("[train] Stopped training at iteration", iteration_ctr, "epochs", epoch_ctr)
    logger.write()
    logger.close()


def main():
    import argparse
    import yaml
    
    
    default_config_path = '/sdf/home/y/youngsam/sw/dune/siren-t/config/siren_4848.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=default_config_path)
    args = parser.parse_args()
    
    
    cfg = yaml.safe_load(open(args.config))
    
    train(cfg)

    


if __name__ == "__main__":
    main()