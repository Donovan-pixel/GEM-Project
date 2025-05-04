import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from src.model import load_model, llama_model_path
from src.data.preprocessing.webnlg import WebNLGGraphTextDataset
from src.utils.collate import graph_llm_collate_fn
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate


def main(args):

    # Step 1: Init W&B and seed
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"webnlg_{args.model_name}_seed{seed}",
               config=args)
    seed_everything(seed=args.seed)

    # Step 2: Load datasets (preprocessed graphs + serialized RDF)
    train_dataset = WebNLGGraphTextDataset(
        graph_dir="dataset/webnlg/train",
        jsonl_path="dataset/train.jsonl",
        split="train")

    val_dataset = WebNLGGraphTextDataset(
        graph_dir="dataset/webnlg/dev",
        jsonl_path="dataset/dev.jsonl",
        split="dev")

    test_dataset = WebNLGGraphTextDataset(
        graph_dir="dataset/webnlg/test",
        jsonl_path="dataset/test.jsonl",
        split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=graph_llm_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=graph_llm_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=graph_llm_collate_fn)

    # Step 3: Load model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)

    # Step 4: Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    trainable_params, all_param = model.print_trainable_params()
    print(f"Trainable: {trainable_params} / {all_param} ({100 * trainable_params / all_param:.2f}%)")

    # Step 5: Training
    num_training_steps = args.num_epochs * len(train_loader)
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr, 'Train Loss': accum_loss / args.grad_steps})
                accum_loss = 0.0

            epoch_loss += loss.item()
            accum_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), accum_loss=accum_loss / (step + 1))

        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({'Train Loss (Epoch)': avg_train_loss})
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

        # Step 6: Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", unit="batch")
            for batch in val_bar:
                val_loss += model(batch).item()
                val_bar.set_postfix(val_loss=val_loss / (val_bar.n + 1))
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({'Val Loss': avg_val_loss})

        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Step 7: Evaluation
    print("Reloading best model...")
    model = _reload_best_model(model, args)
    model.eval()

    os.makedirs(f'{args.output_dir}/webnlg', exist_ok=True)
    out_path = f"{args.output_dir}/webnlg/graphllm_pred_seed{seed}.jsonl"
    print(f"Saving predictions to {out_path}")

    with open(out_path, 'w') as f:
        for batch in tqdm(test_loader, desc="Generating predictions"):
            with torch.no_grad():
                output = model.inference(batch)
                for i in range(len(output['id'])):
                    entry = {
                        'id': output['id'][i],
                        'pred': output['pred'][i],
                        'label': output['label'][i],
                        'desc': output['desc'][i],
                        'question': output['question'][i],
                    }
                    f.write(json.dumps(entry) + '\n')

    # Step 8: Compute metrics
    acc = eval_funcs['webnlg'](out_path)
    print(f"Test Accuracy / BLEU / METEOR: {acc}")
    wandb.log({'Test Metric': acc})


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
