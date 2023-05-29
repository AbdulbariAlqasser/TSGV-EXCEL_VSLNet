import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", type=str, default="ckpt", help="path to save trained model weights"
)
parser.add_argument("--model_name", type=str, default="vslnet", help="model name")
parser.add_argument(
    "--suffix",
    type=str,
    default=None,
    help="set to the last `_xxx` in ckpt repo to eval results",
)
parser.add_argument(
    "--followup_train",
    type=bool,
    default=True,
    help="follow-up training from the last check point",
)
configs = parser.parse_args()
# configs["hi"] = "hi"
print(dir(parser))
