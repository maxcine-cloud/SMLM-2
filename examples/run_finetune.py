# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import time
import math
import argparse
import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_convert_sequenc_id_to_features as convert_sequenc_id_to_features
from transformers import glue_output_modes as output_modes
from  transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}
                    
TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"] 

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
        
def scaled_input(args,emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    if baseline is None:
        baseline = torch.zeros_like(emb)   

    num_points = batch_size * num_batch #2*4
    scale = 1.0 / num_points   #0.08333333333333333
    if start_i is None:
        step = (emb- baseline) * scale #[6, 1, 12, 100, 100]
        res = torch.cat([torch.add(baseline, step*i) for i in range(num_points)], dim=0) #[6, 8, 12, 100, 100]
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]
        
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('******************************parameters size:*************************************', p)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    best_aupr = 0
    last_aupr = 0
    stop_count = 0
    best_loss = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # metr={"aupr": 0.0,
        # "avg_p": 0.0,
        # "avg_u":  0.0,
        # "p_rate": 0.0,
        # "precision": 0.0,
        # "recall": 0.0
        # }
        metr={'acc':0.0, 'auc':0.0,'aupr':0.0, 'f1':0.0, 'mcc':0.0, 'precision':0.0, 'recall':0.0, 'spe':0.0}
        is_zero=0
        if args.task_name[:3] == "dna":
            softmax = torch.nn.Softmax(dim=1)
        
        for step, batch in enumerate(epoch_iterator): #########

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "input_ids2": batch[1], "attention_mask": batch[2], "labels": batch[4],'ref_to_alt_ids': batch[5],'HCF': batch[6]}
            # print('##################',inputs)
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[3] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            t0 = time.time()
            outputs = model(**inputs)
            t1 = time.time()
            train_time=t1-t0
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            train_logits = outputs[1]
            train_preds = train_logits.detach().cpu().numpy()
            if args.output_mode == "classification":
                if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                    if args.do_ensemble_pred:
                        train_probs = softmax(torch.tensor(train_preds, dtype=torch.float32)).numpy()
                    else:
                        train_probs = softmax(torch.tensor(train_preds, dtype=torch.float32))[:,1].numpy()
                elif args.task_name == "dnasplice":
                    train_probs = softmax(torch.tensor(train_preds, dtype=torch.float32)).numpy()
                train_preds = np.argmax(train_preds, axis=1) 
            elif args.output_mode == "regression":
                train_preds = np.squeeze(train_preds)
            if args.do_ensemble_pred:
                train_result = compute_metrics(args.task_name,train_preds , inputs["labels"].detach().cpu().numpy(), train_probs[:,1])
            else:
                train_result = compute_metrics(args.task_name,train_preds , inputs["labels"].detach().cpu().numpy(), train_probs)            
            if  not any(train_result.values()):
                is_zero+=1
            else:
                for key in sorted(train_result.keys()):
                    metr[key]+= np.around(train_result[key],3)
                
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results,eval_loss = evaluate(args, model, tokenizer,global_step)


                        if args.task_name[:3] == "dna":
                            # record the best aupr
                            if (results["aupr"] > best_aupr) or (results["aupr"] == best_aupr and eval_loss < best_loss):
                                best_aupr = results["aupr"]
                                best_loss = eval_loss
                                best_dir=args.output_dir+'/best_res'
                                if not os.path.exists(best_dir) and args.local_rank in [-1, 0]:
                                        os.makedirs(best_dir)
                                logger.info("Saving best model  to %s", best_dir)
                                model_to_save = (
                                            model.module if hasattr(model, "module") else model
                                        )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(best_dir)
                                tokenizer.save_pretrained(best_dir)
                                # Good practice: save your training arguments together with the trained model
                                torch.save(args, os.path.join(best_dir, "training_args.bin"))
                                output_eval_file = os.path.join(best_dir, "best_results.txt")
                                with open(output_eval_file, "w") as writer:
                                    if args.task_name[:3] == "dna":
                                        eval_result = args.data_dir.split('/')[-1] + " " +str(global_step)+ " "+str('')+ " "
                                    else:
                                        eval_result = ""
                                    for key in sorted(results.keys()):
                                        logger.info("  %s = %s", key, str(results[key]))
                                        eval_result = eval_result + str(np.around(results[key],3)) + " " 
                                    eval_result = eval_result+str(np.around(eval_loss,3)) + " "
                                    logger.info(" loss= %s", str(eval_loss))
                                    writer.write(eval_result + "\n")

                        if args.early_stop != 0:
                            # record current aupr to perform early stop
                            if results["aupr"] < last_aupr:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_aupr = results["aupr"]
                            
                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step


                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs["tr_loss"] = tr_loss
                    logs["step"] = step
                    logging_loss = tr_loss
                    output_train_file = os.path.join(args.output_dir, "train_results.txt")
                    logger.info("***** train results {} *****".format(''))
                    with open(output_train_file, "a") as wri:
                        if args.task_name[:3] == "dna":
                            train_result = args.data_dir.split('/')[-1] + " " +str(global_step)+ " "+ str(train_time)+ " " #6 
                        else:
                            train_result = ""
                        for key in sorted(metr.keys()):
                            logger.info("  %s = %s", key, str(np.around((metr[key]/(args.logging_steps-is_zero)),3)))
                            train_result = train_result + str(np.around((metr[key]/(args.logging_steps-is_zero)),3)) + " "
                        train_result=train_result + str(np.around(loss_scalar,3))+ " "
                        logger.info("  loss = %s", str(loss_scalar))                         
                        wri.write(train_result + "\n")
                    is_zero=0
                    for key in sorted(metr.keys()):
                        metr[key]= 0.0                             
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.task_name == "dna690" and results["aupr"] < best_aupr:
                        continue
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if (args.max_steps > 0 and global_step > args.max_steps):
                epoch_iterator.close()
                break
        if (args.max_steps > 0 and global_step > args.max_steps):
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer,step, prefix="", evaluate=True,last_value=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)
        

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "input_ids2": batch[1], "attention_mask": batch[2], "labels": batch[4],'ref_to_alt_ids': batch[5],'HCF': batch[6]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                t2 = time.time()
                outputs= model(**inputs)
                t3 = time.time()
                test_time=t3-t2
                tmp_eval_loss, logits = outputs[:2]
                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logger.info("eval_loss = %f", eval_loss)
        logger.info("nb_eval_steps = %d", nb_eval_steps)
        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if args.do_ensemble_pred:
            if last_value:
                result = compute_metrics(eval_task, preds, out_label_ids, probs[:,1],last_value, is_val=1)
            else:            
                result = compute_metrics(eval_task, preds, out_label_ids, probs[:,1], is_val=1)
        else:
            if last_value:
                result = compute_metrics(eval_task, preds, out_label_ids, probs,last_value, is_val=1)
            else:
                result = compute_metrics(eval_task, preds, out_label_ids, probs, is_val=1)
        results.update(result)
        if args.task_name == "dna690":
            eval_output_dir = args.result_dir
            if not os.path.exists(args.result_dir): 
                os.makedirs(args.result_dir)
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "a") as writer:

            if args.task_name[:3] == "dna":
                eval_result = args.data_dir.split('/')[-1] + " " +str(step)+ " "+str(test_time)+ " "
            else:
                eval_result = ""
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                eval_result = eval_result + str(np.around(result[key],3)) + " " 
            eval_result = eval_result+str(np.around(eval_loss,3)) + " "
            logger.info(" loss= %s", str(eval_loss))
            writer.write(eval_result + "\n")

    if args.do_ensemble_pred:
        return results, eval_loss,eval_task, preds, out_label_ids, probs
    else:
        return results,eval_loss



def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        preds = None
        out_hidden =None        
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "input_ids2": batch[1], "attention_mask": batch[2], "labels": batch[4],'ref_to_alt_ids': batch[5], 'HCF': batch[6]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        if args.do_ensemble_pred:
            result = compute_metrics(pred_task, preds, out_label_ids, probs[:,1])
        else:
            result = compute_metrics(pred_task, preds, out_label_ids, probs)
        
        pred_output_dir = args.predict_dir
        if not os.path.exists(pred_output_dir):
               os.makedir(pred_output_dir)
        output_pred_file = os.path.join(pred_output_dir, "pred_results.txt")
        logger.info("***** Pred results {} *****".format(prefix))
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
        # np.savetxt(output_pred_file, probs,fmt="%.6f")
        with open(output_pred_file, "a") as writer:
            if args.task_name[:3] == "dna":
                eval_result = args.data_dir.split('/')[-1] + "\t" 
            else:
                eval_result = ""
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                eval_result = eval_result + str(np.around(result[key],3)) + "\t" 
            writer.write(eval_result + "\n")
            
def extractHiddenOutpu(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)
        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        hidden_features = None
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "input_ids2": batch[1], "attention_mask": batch[2], "labels": batch[4],'ref_to_alt_ids': batch[5], 'HCF': batch[6]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
            if hidden_features is None:
                hidden_features = hiddenOutput.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                hidden_features = np.append(hidden_features, hiddenOutput.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        pred_output_dir = args.predict_dir
        if not os.path.exists(pred_output_dir):
               os.makedir(pred_output_dir)
        output_pred_file = os.path.join(pred_output_dir, "pred_hiddenOutput.txt")
        np.savetxt(output_pred_file, np.hstack((hidden_features.reshape(-1,768),out_label_ids.reshape(-1,1))),fmt="%.6f")

            
def visualize(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)
    

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''
            
            
        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset),2])
        else:
            preds = np.zeros([len(pred_dataset),3])
        attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
        
        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],"input_ids2": batch[1], "attention_mask": batch[2], "labels": batch[4],'ref_to_alt_ids': batch[5],'seq_id': batch[6]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                attention = outputs[-1][-1]
                _, logits = outputs[:2]

                
                preds[index*batch_size:index*batch_size+len(batch[0]),:] = logits.detach().cpu().numpy()
                attention_scores[index*batch_size:index*batch_size+len(batch[0]),:,:,:] = attention.cpu().numpy()
                # if preds is None:
                #     preds = logits.detach().cpu().numpy()
                # else:
                #     preds = np.concatenate((preds, logits.detach().cpu().numpy()), axis=0)

                # if attention_scores is not None:
                #     attention_scores = np.concatenate((attention_scores, attention.cpu().numpy()), 0)
                # else:
                #     attention_scores = attention.cpu().numpy()
        
        if args.task_name != "dnasplice":
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
        else:
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()

        scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

        for index, attention_score in enumerate(attention_scores):
            attn_score = []
            for i in range(1, attention_score.shape[-1]-kmer+2):
                attn_score.append(float(attention_score[:,0,i].sum()))

            for i in range(len(attn_score)-1):
                if attn_score[i+1] == 0:
                    attn_score[i] = 0
                    break

            # attn_score[0] = 0    
            counts = np.zeros([len(attn_score)+kmer-1])
            real_scores = np.zeros([len(attn_score)+kmer-1])
            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i+j] += 1.0
                    real_scores[i+j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)
            
        
            # print(index)
            # print(real_scores)
            # print(len(real_scores))

            scores[index] = real_scores
        

    return scores, probs



def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # cached_features_file = os.path.join(
    #     args.data_dir,
    #     "cached_{}_{}_{}_{}".format(
    #         "dev" if evaluate else "train",
    #         list(filter(None, args.model_name_or_path.split("/"))).pop(),
    #         str(args.max_seq_length),
    #         str(task),
    #     ),
    # )
    # if args.do_predict:
    #     cached_features_file = os.path.join(
    #     args.data_dir,
    #     "cached_{}_{}_{}".format(
    #         "dev" if evaluate else "train",
    #         str(args.max_seq_length),
    #         str(task),
    #     ),
    # )
    if args.do_predict or args.do_hiddenOutput:
        cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            "test",
            str(args.max_seq_length),
            str(task),
        ),
    ) 
    else:
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        ####import data
        # examples = (
        #     processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        # )   
        if args.do_predict or args.do_hiddenOutput:
            examples = (
                processor.get_test_examples(args.data_dir) )
        else:
            examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        ) 
        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0


        if args.n_process == 1:
            features = convert_examples_to_features(
            examples[0],
            examples[1],
            tokenizer,
            label_list=label_list,
            max_length=max_length,
            output_mode=output_mode,
            pad_on_left=pad_on_left,  # pad on the left for xlnet
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,)
                
        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc/4),1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples[0])/n_proc) #3
            for i in range(1, n_proc+1): ####1,2,3,4,5,6,7,8
                if i != n_proc:
                    indexes.append(len_slice*(i)) 
                else:
                    indexes.append(len(examples[0])) #[0, 3, 6, 9, 12, 15, 18, 21, 24]
           
            results = []
            
            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(examples[0][indexes[i]:indexes[i+1]], examples[1][indexes[i]:indexes[i+1]],tokenizer, max_length, None, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
                print(str(i+1) + ' processor started !')
            
            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())
                    

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    print(all_input_ids.size())
    all_input_ids2 = torch.tensor([f.input_ids2 for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_ref_to_alt_ids = torch.tensor([[f.ref_to_alt_ids] for f in features], dtype=torch.long)
    logger.info("Loading extra sequence-based features")
    HCF = convert_sequenc_id_to_features(np.array([[f.seq_id] for f in features], dtype=str),args.data_dir)
    HCF = torch.from_numpy(HCF).float()
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_ids2, all_attention_mask, all_token_type_ids, all_labels, all_ref_to_alt_ids,HCF)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='./sample_data/ft/6',
        
        # default='/media/bcm/Elements SE/SMLM_Vmodel/V-DNABERT/DNAVERT_V23/examples/sample_data/ft/6',

        type=str,
        # required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default='dna',
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--n_process",
        default=8,
        type=int,
        help="number of processes used for data process",
    )
    parser.add_argument(
        "--should_continue", action="store_true",default=True, help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        # default='/media/bcm/Elements SE/SMLM_Vmodel/V-DNABERT/DNAVERT_V23/examples/6-new-12w-0',    #'./6-new-12w-0',
        default='./6-new-12w-0',
        
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default='dnaprom',
        type=str,
        # required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default='./output',
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    
    # Other parameters
    parser.add_argument(
        "--visualize_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--result_dir",
        default='./result',
        type=str,
        help="The directory where the dna690 and mouse will save results.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--predict_dir",
        default='./output',
        type=str,
        help="The output directory of predicted result. (when do_predict)",
    )
    parser.add_argument(
        "--max_seq_length",
        default=98,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")###############
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")#################
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.") #args.do_hiddenOutput
    parser.add_argument("--do_hiddenOutput", action="store_true", help="Whether to do prediction on the given dataset.") #
    
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.") #,default=1
    parser.add_argument("--visualize_train", action="store_true", help="Whether to visualize train.tsv or dev.tsv.")
    parser.add_argument("--do_ensemble_pred", action="store_true", help="Whether to do ensemble prediction with kmer 3456.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",default=1
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",
    )
    parser.add_argument(
        "--prune_batch_size", default=16, type=int, help="Batch size per GPU/CPU for pruning.",
    )
    parser.add_argument(
        "--prune_num_batch", default=4, type=int, help="Num Batch size per GPU/CPU for pruning.",
    )
    parser.add_argument(
        "--sample_index", default=0, type=int, help="Index of samples for a batch.",
    )
    parser.add_argument(
        "--early_stop", default=1000, type=int, help="set this to a positive integet if you want to perfrom early stop. The model will stop \
                                                    if the auc keep decreasing early_stop times",
    )
    parser.add_argument(
        "--predict_scan_size",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn_dropout", default=0.0, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent*total_steps.")

    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",default=1
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--visualize_models", type=int, default=None, help="The model used to do visualization. If None, use 3456.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


    args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            #raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
            args.should_continue = False
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if not args.do_visualize and not args.do_ensemble_pred:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        config.split = int(args.max_seq_length/512)
        config.rnn = args.rnn
        config.num_rnn_layer = args.num_rnn_layer
        config.rnn_dropout = args.rnn_dropout
        config.rnn_hidden = args.rnn_hidden

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        logger.info('finish loading model')

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.task_name != "dna690":
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result,_ = evaluate(args, model, tokenizer,global_step, prefix=prefix,last_value=1)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]: #args.do_predict
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        logger.info("Predict using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        prediction = predict(args, model, tokenizer, prefix=prefix)

    # Extract hidden output
    if args.do_hiddenOutput and args.local_rank in [-1, 0]: #args.do_predict
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        logger.info("Extract hidden output using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        # prediction = extractHiddenOutpu(args, model, tokenizer, prefix=prefix)


    # Visualize
    if args.do_visualize and args.local_rank in [-1, 0]:
        visualization_models = [6] if not args.visualize_models else [args.visualize_models] #3,4,5,可视化部分模型没有找到

        scores = None
        all_probs = None

        for kmer in visualization_models:
            output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
            #checkpoint_name = os.listdir(output_dir)[0]
            #output_dir = os.path.join(output_dir, checkpoint_name)
            
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                checkpoint,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            attention_scores, probs = visualize(args, model, tokenizer, prefix=prefix, kmer=kmer)
            if scores is not None:
                all_probs += probs
                scores += attention_scores
            else:
                all_probs = deepcopy(probs)
                scores = deepcopy(attention_scores)

        all_probs = all_probs/float(len(visualization_models))
        np.save(os.path.join(args.predict_dir, "atten.npy"), scores)
        np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # ensemble prediction
    if args.do_ensemble_pred and args.local_rank in [-1, 0]:

        for kmer in range(3,7):
            output_dir = os.path.join(args.output_dir, str(kmer))
            tokenizer = tokenizer_class.from_pretrained(
                "dna"+str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            if kmer == 3:
                args.data_dir = os.path.join(args.data_dir, str(kmer))
            else:
                args.data_dir = args.data_dir.replace("/"+str(kmer-1), "/"+str(kmer))

            if args.result_dir.split('/')[-1] == "test.npy":
                results,_, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix)
            elif args.result_dir.split('/')[-1] == "train.npy":
                results,_, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False)
            else:
                raise ValueError("file name in result_dir should be either test.npy or train.npy")

            if kmer == 3:
                all_probs = deepcopy(probs)
                cat_probs = deepcopy(probs)
            else:
                all_probs += probs
                cat_probs = np.concatenate((cat_probs, probs), axis=1)
            print(cat_probs[0])
        

        all_probs = all_probs / 4.0
        all_preds = np.argmax(all_probs, axis=1)
        
        # save label and data for stuck ensemble
        labels = np.array(out_label_ids)
        labels = labels.reshape(labels.shape[0],1)
        data = np.concatenate((cat_probs, labels), axis=1)
        random.shuffle(data)
        root_path = args.result_dir.replace(args.result_dir.split('/')[-1],'')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        np.save(args.result_dir, data)
        ensemble_results = compute_metrics(eval_task, all_preds, out_label_ids, all_probs[:,1])
        logger.info("***** Ensemble results {} *****".format(prefix))
        for key in sorted(ensemble_results.keys()):
            logger.info("  %s = %s", key, str(ensemble_results[key]))    


            


    return results


if __name__ == "__main__":
    main()
