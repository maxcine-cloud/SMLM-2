# SMLM-2

​	We developed a deep biological language learning model SMLM-2, based on DNABERT (Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021;, btab083, https://doi.org/10.1093/bioinformatics/btab083). The common information of DNA sequence pre-trained language model is transferred to modeling synonymous mutations. SMLM-2 is a framework trained on biological sequence data combined with hand-craft features of epSMic-2, aiming to measure the impact of synonymous mutations by learning potential feature space representations. 

​	We provides source codes of SMLM-2 based on the DNABERT model and usage examples. Pre-trained and fine-tuned models are derived from the DNABERT. 

## Environment setup

Please refer to DNABERT (Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021, btab083, https://doi.org/10.1093/bioinformatics/btab083) of the detailed python virtual environment set with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). 

- python 3.6.13
- scikit-learn 0.24.2
- sentencepiece 0.1.91
- tensorboardX 2.5.1
- tensorboard 2.10.0
- pandas 1.1.5

To create a conda environment for SMLM-2, you can import the entire environment for:

```
conda env create -f environment.yml
```

#### Install the requirements

```
cd DNABERT
python3 -m pip install --editable .
```

(Optional, install apex for fp16 training)

change to a desired directory by `cd PATH_NAME`

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Fine-tune 

#### Data processing

Please see the template data at `/example/sample_data/ft/`. If you are trying to fine-tune DNABERT (Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021, btab083, https://doi.org/10.1093/bioinformatics/btab083) with your own data, please process you data into the same format as it. The sequences are in kmer format.

#### Download pre-trained DNABERT

Please refer to DNABERT (Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021, btab083, https://doi.org/10.1093/bioinformatics/btab083) for downloading the pre-trained language model, and then place it in the example folder. Alternatively, you can directly click on these links for access. 

[DNABERT3](https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view?usp=sharing)

[DNABERT4](https://drive.google.com/file/d/1V7CChcC6KgdJ7Gwdyn73OS6dZR_J-Lrs/view?usp=sharing)

[DNABERT5](https://drive.google.com/file/d/1KMqgXYCzrrYD1qxdyNWnmUYPtrhQqRBM/view?usp=sharing)

[DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)

We also provide a model with `KMER=3`  of DNABERT at `/examples/`.

#### Fine-tune with pre-trained model

In the following example,  we use SMLM-2 with kmer=3 as example. 

```
cd examples

export KMER=3
export MODEL_PATH=./3-new-12w-0
export DATA_PATH=sample_data/ft/$KMER
export OUTPUT_PATH=./output
  
python run_finetune.py \
    --model_type dna \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 101 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output_dir \
    --weight_decay 0.01 \	
    --n_process 72
```



## Prediction

After the model is fine-tuned, we can get predictions by running

```$
cd examples

export KMER=3
export MODEL_PATH=./3-new-12w-0
export DATA_PATH=sample_data/ft/$KMER
export OUTPUT_PATH=./output

python run_finetune.py \
    --model_type dna \
    --model_name_or_path $OUTPUT_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 101 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output_dir \
    --weight_decay 0.01 \	
    --n_process 72
```

 

## Citation

TO DO