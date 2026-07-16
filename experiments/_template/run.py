# experiments/_template/run.py
import argparse, os, json, logging, sys
from datetime import datetime


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("experiment")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # 1) Make output folders
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "attention_maps"), exist_ok=True)
    log = setup_logging(args.output)
    log.info("Starting experiment -> %s", args.output)

    # 2) Read config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    log.info("Config: %s", config)

    # 3) === YOUR CODE HERE ===
    # from src.diffusion_lens import run_diffusion_lens
    # images, attention_maps, metrics = run_diffusion_lens(config)
    # save each image to args.output/images/, each map to args.output/attention_maps/
    # (this is where FLUX.2 Klein + Qwen actually run on the GPU)

    # 4) Save a metrics summary
    metrics = {
        "name": config.get("name"),
        "prompt": config.get("prompt"),
        "status": "ok",
        "finished_at": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Done. Everything saved to %s", args.output)


if __name__ == "__main__":
    main()
