import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')

        # Detect CPU vs GPU
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if is_cuda else 'cpu')
        dtype = torch.float16 if is_cuda else torch.float32
        device_map = "auto" if is_cuda else None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path,
            use_fast=False,
            revision="main"
        )
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            revision="main"
        )
        print(f"LLAMA loaded on device {device} with dtype {dtype}")

        # Freeze or apply LoRA
        if args.llm_frozen == 'True':
            print("Freezing LLAMA weights!")
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("Applying LORA to LLAMA!")
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model

        # Graph encoder
        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            hidden_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(device)

        # Project graph embeddings to match LLM input space
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return next(self.parameters()).device

    def maybe_autocast(self, dtype=torch.float16):
        return autocast(dtype=dtype) if self.device.type == 'cuda' else contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples.to(self.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index, graphs.edge_attr)
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds

    def forward(self, samples):
        questions = self.tokenizer(
            ["Generate a natural language sentence that describes the following RDF graph:"] * len(samples.id),
            add_special_tokens=False
        )
        descriptions = self.tokenizer(samples.desc, add_special_tokens=False)
        labels = self.tokenizer(samples.label, add_special_tokens=False)

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        graph_embeds = self.projector(self.encode_graphs(samples))

        batch_size = len(samples.id)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            batch_label_input_ids.append([IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids)

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_len + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        questions = self.tokenizer(
            ["Generate a natural language sentence that describes the following RDF graph:"] * len(samples.id),
            add_special_tokens=False
        )
        descriptions = self.tokenizer(samples.desc, add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        graph_embeds = self.projector(self.encode_graphs(samples))

        batch_size = len(samples.id)
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'id': samples.id,
            'pred': pred,
            'label': samples.label,
            'question': samples.question,
            'desc': samples.desc,
        }

    def print_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in self.parameters())
        return trainable_params, all_param