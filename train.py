import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from logging import getLogger

import numpy as np
import torch

from src.datasets import CharDataset, InfiniteDataLoader, load_initial_data, update_datasets
from src.envs import ENVS, build_env
from src.envs.environment import do_stats
from src.evaluator import sample_and_score
from src.models.model import Transformer
from src.trainer import reload_model_optimizer, train
from src.utils import bool_flag, force_release_memory, initialize_exp, load_run_state, log_resources, save_run_state, write_important_metrics

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser("A simple Axplorer loop for different maths problems")

    parser.add_argument("--gensize", type=int, default=100000, help="Number of generate initial values")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=50000, help="number of training steps.")
    parser.add_argument("--num_samples_from_model", type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument("--pop_size", type=int, default=200000, help="Total maximum number of examples at each epoch")
    parser.add_argument("--ntest", type=int, default=1000, help="Size of test set")
    parser.add_argument("--env_name", type=str, default="square", help="Math problem to be addressed")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    parser.add_argument("--process_pool", type=bool_flag, default="true", help="use process_pool to generate and score initial data")
    parser.add_argument("--always_search", type=bool_flag, default="true", help="if True, use local search for all examples generated")
    parser.add_argument("--redeem_only", type=bool_flag, default="false", help="if True, save invalid examples only")
    parser.add_argument("--new_proportion", type=float, default=0.0, help="proportion of new samples in test set")

    parser.add_argument("--num_workers", type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument("--num_eval_steps", type=int, default=500, help="number of step between each evaluation during training.")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    # sampling
    parser.add_argument("--top_k", type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument("--n_layer", type=int, default=4, help="number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads (in a transformer)")
    parser.add_argument("--n_embd", type=int, default=256, help="number of feature channels in the model")
    parser.add_argument("--no_positional", type=bool_flag, default="false", help="no positional embedding")
    parser.add_argument("--max_len", type=int, default=500, help="Block size, maximum length of sequences")

    # optimization
    parser.add_argument("--batch_size", type=int, default=32, help="batch size during optimization")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument("--gen_batch_size", type=int, default=1000, help="generation batch size")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--temp_span", type=int, default=0, help="temperature span")
    parser.add_argument("--inc_temp", type=float, default=0.0, help="temperature")
    parser.add_argument("--keep_only_unique", type=bool_flag, default="true", help="keep only unique data")
    parser.add_argument("--save_best", type=bool_flag, default="false", help="save best model based on test loss")

    # path and ports
    parser.add_argument("--dump_path", type=str, default="checkpoint", help="Experiment dump path")
    parser.add_argument("--command", type=int, default=1, help="Command 1 to allow writing on metrics.csv")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cpu", type=bool_flag, default="false", help="run on cpu only")
    parser.add_argument("--data_generation_only", type=bool_flag, default="false", help="only generate data and exit")

    return parser


if __name__ == "__main__":
    print("Starting training script")
    parser = get_parser()
    args = parser.parse_args()

    if args.exp_id == "" and os.environ.get("MODAL_EXP_ID") is None:
        os.environ["MODAL_EXP_ID"] = time.strftime("%Y_%m_%d_%H_%M_%S")
        args.exp_id = os.environ["MODAL_EXP_ID"]

    args.device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cuda")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    fused = True if args.device in ["cuda", "mps"] else False

    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")
    print(f"Starting with seed: {args.seed}")

    env = build_env(args)
    exists_optimum = env.exists_optimum
    # counters of the whole run: reloaded from run_state.json when a run is restarted
    # always defined, so the final log works also when no optimum is known
    run_state = load_run_state(
        args.dump_path, {"optimum_counter": 0, "first_found": False, "first_found_epoch": None, "n_local_searches": 0, "best_loss": None}
    )
    optimum_counter = run_state["optimum_counter"]
    first_found = run_state["first_found"]
    first_found_epoch = run_state["first_found_epoch"]

    classname = env.data_class

    try:
        # system inits
        torch.manual_seed(args.seed)

        args.vocab_size = len(env.tokenizer.itos)

        args.block_size = args.max_len + 2
        stoi = env.tokenizer.stoi
        itos = env.tokenizer.itos

        # Initialize transformer
        model = Transformer(args, stoi["PAD"], stoi["EOS"])
        model.to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8, fused=fused
        )
        reload_model_optimizer(args, model, optimizer)

        print(f"Loading initial data for {classname}")
        train_set, test_set = load_initial_data(args, classname)
        if args.data_generation_only:
            logger.info("Data generation only mode. Exiting...")
            exit(0)
        train_data_path = os.path.join(args.dump_path, "train_data.pkl")
        test_data_path = os.path.join(args.dump_path, "test_data.pkl")

        # log initial stats
        metrics = do_stats(-1, data=train_set)
        temperature = args.temperature

        # Loop of Axplorer
        best_loss = run_state["best_loss"]
        epoch_file = os.path.join(args.dump_path, "epoch.txt")
        if os.path.isfile(epoch_file):
            with open(epoch_file, "r") as f:
                n_epoch = int(f.read())
        else:
            n_epoch = 0
        temp_file = os.path.join(args.dump_path, "temperature.txt")
        if os.path.isfile(temp_file):
            with open(temp_file, "r") as f:
                temperature = float(f.read())
        else:
            temperature = args.temperature

        metric_file = os.path.join(args.dump_path, "metrics.csv")
        # which important metrics here?
        if os.path.isfile(metric_file) and os.path.getsize(metric_file) > 0:
            # restarted run: the initial row is already in metrics.csv, only record the command
            with open(os.path.join(args.dump_path, "command.txt"), "a") as f:
                f.write(f"{args.command}\n")
        else:
            write_important_metrics(metrics, n_epoch, metric_file, command=args.command)

        n_local_searches = run_state["n_local_searches"]

        for epoch in range(n_epoch, args.max_epochs):
            logger.info(f"==== Starting Epoch {n_epoch} =====")
            log_resources(f"Epoch {epoch} START")
            print(f"==== Starting Epoch {n_epoch} =====")

            if args.device == "cuda":
                torch.cuda.empty_cache()
            elif args.device == "mps":
                torch.mps.empty_cache()

            # tokenize
            print("Tokenizing data")
            train_words = [env.tokenizer.encode(d) for d in train_set]
            test_words = [env.tokenizer.encode(d) for d in test_set]
            # data loaders
            train_dataset = CharDataset(train_words, args.max_len, stoi)
            test_dataset = CharDataset(test_words, args.max_len, stoi)
            force_release_memory()
            # -- Data used for training, so local search must been applied before! --

            if args.device == "cuda":
                logger.info(
                    f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB"
                )
            elif args.device == "mps":
                logger.info(
                    f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB"
                )

            batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=args.device == "cuda", num_workers=0)
            try:
                print(f"Training model for {args.num_eval_steps} steps with batch size {args.batch_size}")
                best_loss, epoch_flops = train(model, args, batch_loader, optimizer, test_dataset, current_best_loss=best_loss)
            finally:
                batch_loader.close()
                del batch_loader
            log_resources(f"Epoch {epoch} AFTER_TRAIN")
            force_release_memory()

            logger.info(f"Sample with temperature {temperature} to {temperature+0.1*args.temp_span}")
            print(f"Sample with temperature {temperature} to {temperature+0.1*args.temp_span}")
            if args.device == "cuda":
                torch.cuda.empty_cache()
            elif args.device == "mps":
                torch.mps.empty_cache()

            # AFTER TRAIN: sample_and_score -> genera nuovi grafi con il modello allenato
            # Search & Score phase, BUT ALSO IMPROVEMENTE:
            # sample_and_score -> do_score -> _do_score -> local_search
            # len(new_data) dovrebbe essere lo stesso ad ogni epoca
            print(f"Sampling and scoring {args.num_samples_from_model} examples")
            new_data, search_stats = sample_and_score(model, args, stoi, itos, env, temperature, args.temp_span)
            log_resources(f"Epoch {epoch} AFTER_SAMPLE")
            n_local_searches += search_stats["num_local_searches"]

            if args.device == "cuda":
                torch.cuda.empty_cache()
            elif args.device == "mps":
                torch.mps.empty_cache()

            # Possible to add another generation method here and mix it before taking the best
            # Train e test set vengono aggiornati (SOVRASCRITTI)
            train_set, test_set, inc_temp = update_datasets(args, new_data, train_set, test_set, train_data_path, test_data_path)
            log_resources(f"Epoch {epoch} AFTER_UPDATE_DATASETS")
            force_release_memory()

            # Possible to add another generation method here and mix it before taking the best
            if inc_temp and args.inc_temp > 0.0:
                temperature += args.inc_temp

            # here metric is csv, so I can't add columns for optimum, but I can log it
            print("Doing stats on train set")
            metrics = do_stats(-1, data=train_set)
            if metrics is not None:
                metrics.update(search_stats)
                metrics["flops"] = epoch_flops
            # here i can check optimum?
            if exists_optimum and env.optimum is not None:
                if metrics["max"] >= env.optimum and not first_found:
                    # Il calcolo delle ricerche locali è corretto?
                    logger.info(f"Optimum {env.optimum} reached at epoch {n_epoch} AFTER {n_local_searches} local searches")
                    n_opt = sum(1 for d in new_data if d.score == env.optimum)
                    optimum_counter += n_opt
                    first_found = True
                    first_found_epoch = n_epoch

            n_epoch += 1
            # saved before epoch.txt: a restart never counts this epoch twice
            save_run_state(
                args.dump_path,
                {
                    "optimum_counter": optimum_counter,
                    "first_found": first_found,
                    "first_found_epoch": first_found_epoch,
                    "n_local_searches": n_local_searches,
                    "best_loss": best_loss,
                },
            )
            with open(epoch_file, "w") as f:
                f.write(str(n_epoch))
            with open(temp_file, "w") as f:
                f.write(str(temperature))

            print(f"Writing metrics to {metric_file}")
            write_important_metrics(metrics, n_epoch, metric_file)

            if n_epoch >= args.max_epochs:
                # Forse non serve salvare n_epochs quì, potrebbe essere ridondante
                logger.info(f"Stopping training. Number of optimums found: {optimum_counter} | Number of epochs: {n_epoch} | Number of local searches: {n_local_searches} | First optimum at epoch: {first_found_epoch}.")
                print(f"Stopping training. Number of optimums found: {optimum_counter} | Number of epochs: {n_epoch} | Number of local searches: {n_local_searches} | First optimum at epoch: {first_found_epoch}.")
                break
    except Exception:
        logger.exception("Training crashed with an unhandled exception")
        if args.device == "cuda" and torch.cuda.is_available():
            logger.error(
                f"CUDA memory at crash: allocated={torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, "
                f"reserved={torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB, "
                f"max_allocated={torch.cuda.max_memory_allocated(0)/(1024*1024):.2f}MB"
            )
        raise
