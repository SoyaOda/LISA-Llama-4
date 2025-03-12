import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, AutoTokenizer

from model.LISA import LISAForCausalLM
from model.llama3_2 import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="meta-llama/Llama-3.2-11B-Vision-Instruct", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llama_3",
        type=str,
        choices=["llava_v1", "llava_llama_2", "llama_3"],
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_interval", default=10, type=int)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(args.version)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    # <SEG>トークンを追加
    special_tokens = {"additional_special_tokens": ["<SEG>"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # プロセッサのトークナイザにも同じトークンを追加
    if processor is not None and hasattr(processor, "tokenizer"):
        processor.tokenizer.add_special_tokens(special_tokens)
    
    # <SEG>トークンのIDを保存
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("<SEG>")
    print(f"Added <SEG> token with ID: {args.seg_token_idx}")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        # プロセッサのトークナイザにも追加
        if processor is not None and hasattr(processor, "tokenizer"):
            processor.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM(
        model_id=args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=args.train_mask_decoder,
        out_dim=args.out_dim,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        seg_token_idx=args.seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        vision_tower=args.vision_tower,
        use_mm_start_end=args.use_mm_start_end,
        # DeepSpeed環境ではdevice_mapは使用しない
        device_map=None
    )

    # DeepSpeedの初期化前にモデルの一部をCPUに移動
    # これにより、metaデバイスのテンソルのコピーエラーを回避
    if hasattr(model.model, "to"):
        try:
            print("モデルをCPUに移動します（meta device回避）")
            # まず主要なモデルをCPUに移動
            model.model.to("cpu")
        except Exception as e:
            print(f"警告: モデルのCPU移動中にエラーが発生しました: {e}")
            print("DeepSpeedが自動的に処理するため続行します")
    
    # make text_hidden_fcs, mask_decoder trainable
    for n, p in model.named_parameters():
        if any([x in n for x in ["mask_decoder", "text_hidden_fcs"]]):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
            
    # 埋め込みの調整 - LISAForCausalLMでは既に行われているが、念のためここでも行う
    # トークン埋め込みのサイズ変更（入力と出力の両方）
    model.resize_token_embeddings(len(tokenizer))
    
    # 入力埋め込みと出力埋め込みを結合
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if hasattr(model.model, "tie_weights"):
        model.model.tie_weights()
    
    if hasattr(model.model, "config"):
        model.model.config.eos_token_id = tokenizer.eos_token_id
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()

    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        args.conv_type, conversation_lib.conv_templates["llama_3"]
    )

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # PEFT適用前にconfigがPretrainedConfigオブジェクトであることを確認
        # 辞書型の場合はPretrainedConfigに変換
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            if isinstance(model.model.config, dict):
                print("警告: model.model.configが辞書型です。PretrainedConfigに変換します。")
                from transformers import PretrainedConfig
                config_dict = model.model.config.copy()
                model.model.config = PretrainedConfig()
                for key, value in config_dict.items():
                    setattr(model.model.config, key, value)
                
                # model_type属性を追加
                if not hasattr(model.model.config, 'model_type'):
                    print("model.model.configにmodel_type属性を追加します")
                    model.model.config.model_type = "mllama"
        
        # 元のconfigオブジェクトの保持を確認
        print(f"PEFT適用前のモデル設定確認:")
        print(f"  - model.config type: {type(model.config).__name__}")
        if hasattr(model, 'model'):
            print(f"  - model.model.config type: {type(model.model.config).__name__}")
            if hasattr(model.model.config, 'model_type'):
                print(f"  - model.model.config.model_type: {model.model.config.model_type}")
        
        # PEFTモデルの作成
        model = get_peft_model(model, lora_config)
        
        # PEFTモデル作成後、base_modelのconfigを検証
        print(f"PEFT適用後のモデル設定確認:")
        print(f"  - model.base_model.config type: {type(model.base_model.config).__name__}")
        if hasattr(model.base_model.config, 'model_type'):
            print(f"  - model.base_model.config.model_type: {model.base_model.config.model_type}")
            
        model.print_trainable_parameters()

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    # 小規模テストデータセットを使用する場合の警告
    if "small_test_dataset" in args.dataset_dir:
        print("\n")
        print("=" * 80)
        print("  警告: small_test_datasetを使用しています")
        print("  このデータセットは限られた画像ファイルのみを含んでいます:")
        print("    - ade20k: 10個の画像")
        print("    - coco/cocostuff: 10個の画像")
        print("    - refer_seg/images/mscoco/images/train2014: 9個の画像")
        print("    - reason_seg/ReasonSeg/train: 10組の画像-JSONペア")
        print("  指定された`--dataset`と`--refer_seg_data`がこのデータセットと互換性があることを確認してください。")
        print("=" * 80)
        print("\n")
    
    try:
        train_dataset = HybridDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            processor=processor,
        )
    except FileNotFoundError as e:
        print("\n")
        print("=" * 80)
        print(f"エラー: データセットの初期化に失敗しました - ファイルが見つかりません")
        print(f"原因: {e}")
        print("解決策:")
        print("1. `--dataset`引数と`--refer_seg_data`引数が`small_test_dataset`の構成と互換性があることを確認してください")
        print("2. 以下のコマンドラインオプションを試してください:")
        print("   --dataset=\"sem_seg||vqa||reason_seg\" --sample_rates=\"4,3,1\"")
        print("   または")
        print("   --dataset=\"sem_seg||reason_seg\" --sample_rates=\"4,1\"")
        print("=" * 80)
        print("\n")
        raise
    except Exception as e:
        print("\n")
        print("=" * 80)
        print(f"エラー: データセットの初期化に失敗しました")
        print(f"原因: {e}")
        print("解決策:")
        print("1. データセットの構造とコード実装の互換性を確認してください")
        print("2. `--dataset`および関連するデータセット引数を確認してください")
        print("=" * 80)
        print("\n")
        raise

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
            processor=processor,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    
    # DeepSpeedの初期化前に注意事項を表示
    print("DeepSpeedの初期化を開始します（meta tensorがある場合にエラーが発生する可能性があります）")
    print("問題が発生した場合は、モデルをCPUに完全に移動してからDeepSpeedを初期化してください")
    
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
                processor=processor,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            # マスクデータのデバッグ情報を追加
            if args.debug and global_step % args.debug_interval == 0 and i == 0:
                print(f"\n[マスクデバッグ情報] エポック {epoch}, ステップ {global_step}")
                if "masks_list" in input_dict:
                    masks_list = input_dict["masks_list"]
                    print(f"masks_list長さ: {len(masks_list)}")
                    null_mask_indices = [idx for idx, m in enumerate(masks_list) if m is None]
                    if null_mask_indices:
                        print(f"[マスクNull原因] Noneマスクのインデックス: {null_mask_indices}")
                        
                        # 対応する画像情報も表示
                        if "image_paths" in input_dict:
                            print("対応する画像パス:")
                            for idx in null_mask_indices:
                                if idx < len(input_dict["image_paths"]):
                                    print(f"  インデックス {idx}: {input_dict['image_paths'][idx]}")
                
                # 最初の数個のサンプルの詳細情報を表示
                if "masks_list" in input_dict and len(input_dict["masks_list"]) > 0:
                    for idx in range(min(3, len(input_dict["masks_list"]))):
                        mask = input_dict["masks_list"][idx]
                        if mask is not None:
                            print(f"マスク {idx}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min().item()}, max={mask.max().item()}")
                        else:
                            print(f"マスク {idx}: None")

            try:
                # input_dictにtokenizerを追加
                input_dict["tokenizer"] = tokenizer
                output_dict = model(**input_dict)
            except Exception as e:
                print(f"\n[エラー情報] モデル実行中にエラーが発生しました: {e}")
                print("入力データの情報:")
                for k, v in input_dict.items():
                    try:
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                        elif isinstance(v, list):
                            print(f"  {k}: list of {len(v)} items")
                            if len(v) > 0 and v[0] is not None:
                                first_item = v[0]
                                if isinstance(first_item, torch.Tensor):
                                    print(f"    先頭アイテム: shape={first_item.shape}, dtype={first_item.dtype}")
                                else:
                                    print(f"    先頭アイテム: type={type(first_item)}")
                    except:
                        print(f"  {k}: 情報の取得に失敗")
                
                # エラー内容の詳細表示
                import traceback
                traceback.print_exc()
                
                # 致命的なエラーの場合は再発生
                raise

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            # input_dictにtokenizerを追加
            input_dict["tokenizer"] = tokenizer
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
