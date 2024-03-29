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
from sklearn import metrics
try:
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    # from .metrics import *
    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
    
    def acc_f1_mcc(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(labels, preds)
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc
        }

    def acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "aupr": aupr,
            "precision": precision,
            "recall": recall,
        }

    def acc_f1_mcc_auc_aupr_pre_rec2(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds, average="macro")
        recall = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
        aupr = average_precision_score(labels, probs, average="macro")
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        specificity=tn/(tn+fp)
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "aupr": aupr,
            "precision": precision,
            "recall": recall,
            "spe":specificity,
        }       
    
    def metr1_2(pred,y_test,pred_prob):
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
        ACC=metrics.accuracy_score(y_test,pred) 
        MCC=metrics.matthews_corrcoef(y_test,pred)
        recall=metrics.recall_score(y_test, pred)
        Pre = metrics.precision_score(y_test, pred) 
        specificity=tn/(tn+fp)
        BACC=(recall+specificity)/2.0
        F1_Score=metrics.f1_score(y_test, pred)
        fpr,tpr,threshold=metrics.roc_curve(y_test,pred_prob) 
        roc_auc=metrics.auc(fpr,tpr)
        precision_prc, recall_prc, _ = metrics.precision_recall_curve(y_test, pred_prob) 
        PRC = metrics.auc(recall_prc, precision_prc)
        return {
                "acc": ACC,
                "f1": F1_Score,
                "mcc": MCC,
                "auc": roc_auc,
                "aupr": PRC,
                "precision": Pre,
                "recall": recall,
                "spe":specificity,
            }     
    
    def pu_f1_aupr_pre_rec(preds, labels, probs,last_eval,is_val):
        aupr, avg_p, avg_u, p_rate, precision, recall = evaluate_model(preds, labels, probs,last_eval, is_val)
        return {
            "aupr": aupr,
            "avg_p": avg_p,
            "avg_u": avg_u,
            "p_rate":p_rate,
            "precision":precision,
            "recall":recall,
        }  
        
    def acc_f1_mcc_auc_pre_rec(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds, average="macro")
        recall = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels, probs=None, last_eval=0, is_val=0):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name in ["dna690", "dnapair"]:
            return acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs)
        elif task_name == "dnaprom":
            # return pu_f1_aupr_pre_rec(preds, labels, probs, last_eval, is_val)
            # return acc_f1_mcc_auc_aupr_pre_rec2(preds, labels, probs)
            return metr1_2(preds, labels, probs)
            #return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
            # return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "dnasplice":
            return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
