import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
import torch
from omegaconf import OmegaConf
import clip
from tqdm import tqdm

##############
from llama_inference.llama import Tokenizer
import util.misc as misc





def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=2048, metavar="LENGTH", help="the maximum sequence length")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):


    texts = []
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    

    ###Load CLIP
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)

    ###Load LLM
    llama_model_path = args.llama_model_path
    from llama_inference.llama.generation_text import Llama
    generator = Llama.build(
        ckpt_dir=llama_model_path,
        tokenizer_path=llama_model_path + "/tokenizer.model",
        max_seq_len=200,
        max_batch_size=80,
    )
    

    texts = []
    local_vocabularies = []
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    ###Iterating each Subwords
    for i in range(0, 32000):
        cur_token = tokenizer.decode(i)
        local_vocabularies.append(cur_token)

        ###Generating Bigrams
        prompts = ["a photo of %s"%str(cur_token)]
        results = generator.text_completion(
            prompts,
            max_gen_len=1,
            temperature=0,
            top_p=1.0,
        )

        ###Generating Trigrams
        prompts = ["a photo of %s"%str(cur_token+results[0]['generation'])]
        results_2 = generator.text_completion(
            prompts,
            max_gen_len=1,
            temperature=0,
            top_p=1.0,
        )
        
        ###Save Vocabulary
        cur_cell = {"1": cur_token, "2":results[0]['generation'], "3": results_2[0]['generation']}
        texts.append(cur_cell)

    ##Global Vocabulary
    np.save("Subword_Bigram_Trigram_Vocabulary.npy", texts)

    ##Local Vocabulary
    np.save("local_vocabulary.npy", local_vocabularies)

    
if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
