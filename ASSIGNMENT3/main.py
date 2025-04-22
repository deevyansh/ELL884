#!/usr/bin/env python3
import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import transformers
from tqdm import tqdm
from datasets import Dataset
import random
import pandas as pd
import sacrebleu
from sklearn.metrics import accuracy_score
from typing import List
# from sacrebleu.metrics import BLEU

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class GECConfig:
    """Configuration for the GEC model."""

    output_dir: str = "./gec_model_outputs"
    cache_dir: str = "./cache"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    entry_number = "2022EE31883"
    batch_correct_batch_size: int = 32

class M2Parser:
    """Parser for M2 formatted GEC data."""

    @staticmethod
    def parse_m2_file(filename: str) -> List[Dict]:
        """
        Parse an M2 file into a list of sentence dictionaries.

        Args:
            filename: Path to M2 file

        Returns:
            List of dictionaries with source and target sentences
        """
        data = []
        current_sentence = {}
        source_sentence = None
        corrections = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('S '):
                    if source_sentence is not None and corrections:
                        current_sentence = {
                            'source': source_sentence,
                            'corrections': corrections
                        }
                        data.append(current_sentence)

                    source_sentence = line[2:]
                    corrections = []

                elif line.startswith('A '):
                    if "noop" in line:
                        continue
                    parts = line[2:].split("|||")
                    if len(parts) >= 3:
                        start_idx = int(parts[0].split()[0])
                        end_idx = int(parts[0].split()[1])
                        error_type = parts[1]
                        correction = parts[2]
                        corrections.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'error_type': error_type,
                            'correction': correction
                        })

        if source_sentence is not None and corrections:
            current_sentence = {
                'source': source_sentence,
                'corrections': corrections
            }
            data.append(current_sentence)
        print("length of data",len(data))


        return data[0:5]

    @staticmethod
    def apply_corrections(source: str, corrections: List[Dict]) -> str:
        """
        Apply corrections to a source sentence.

        Args:
            source: Source sentence
            corrections: List of correction dictionaries

        Returns:
            Corrected sentence
        """

        tokens = source.split()
        sorted_corrections = sorted(corrections, key=lambda x: (x['start_idx'], x['end_idx']), reverse=True)

        for correction in sorted_corrections:
            start_idx = correction['start_idx']
            end_idx = correction['end_idx']
            corrected_text = correction['correction']

            if start_idx < len(tokens):
                del tokens[start_idx:end_idx]

                if corrected_text.strip():
                    corrected_tokens = corrected_text.split()
                    for i, token in enumerate(corrected_tokens):
                        tokens.insert(start_idx + i, token)

        corrected_sentence = ' '.join(tokens)

        return corrected_sentence


class GECorrector:
    """GEC system using the BART model."""

    def __init__(self, config: GECConfig):
        """
        Initialize the GEC system.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        self.P1= M2Parser()
        self.PREFIX="Grammar Correction : "
        


    def evaluate(self, source_file: str, reference_file: str):
        """
        Evaluate model using BLEU and exact match.

        Args:
            source_file: Path to input source sentences (uncorrected)
            reference_file: Path to target reference sentences (corrected)

        Returns:
            dict with 'bleu' and 'exact_match' scores
        """
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        
        assert len(sources) == len(references), "Mismatch between number of source and reference sentences."

        predictions = self.batch_correct(sources)

        # Compute BLEU using sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references]).score

        # Compute exact match (sentence-level)
        exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
        exact_score = sum(exact_matches) / len(exact_matches)

        return {
            "bleu": bleu,
            "exact_match": exact_score
        }
        


    def evaluate1(self,eval_prediction):
        import evaluate
        import sacrebleu
        import numpy as np


        exact_match = evaluate.load("exact_match")

        prediction,labels=eval_prediction
        print("Hello, i am in evaluation", prediction.shape,labels.shape )
        decoded_preds=self.tokenizer.batch_decode(prediction,skip_special_tokens=True)
        print(decoded_preds)
        # labels=np.where(labels!=-100 ,labels,self.tokenizer.pad_token_id)
        decoded_labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
        print(decoded_labels)

        em=exact_match.compute(predictions=decoded_preds,references=decoded_labels)["exact_match"]

        bleu=sacrebleu.corpus_bleu(decoded_preds,[decoded_labels])
        bleu_score=bleu.score

        return {"exact_match",em, "bleu_score",bleu_score}


    def preprocess(self,example):
        input_text=self.PREFIX+example["source"]
        target_text =self.P1.apply_corrections(example["source"], example["corrections"])
        model_inputs = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
        labels=self.tokenizer(target_text,truncation=True,padding="max_length",max_length=128)
        model_inputs["labels"]=labels["input_ids"]

        return model_inputs


    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        MODEL_NAME="t5-base"
        
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model=T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)

        ## Now i want to train this model to the fullest
        logger.info("Starting training")
        # implement your code


        train_dataset=Dataset.from_list(train_dataset)
        val_dataset=Dataset.from_list(val_dataset)
        tokenized_train=train_dataset.map(self.preprocess, batched=False)
        tokenized_val=val_dataset.map(self.preprocess,batched=False)

        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # Get PEFT model
        peft_model = get_peft_model(self.model, lora_config)

        # for parameters in peft_model.parameters():
        #   parameters.requires_grad=True

        # Training Arguments for Seq2Seq
        peft_training_args = TrainingArguments(
            output_dir= args.model_path if args.model_path else "./gec_model_outputs",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=1e-4,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_steps=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            label_names=["labels"]
        )

        # Initialize Seq2Seq Trainer
        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer  # Provide tokenizer for proper sequence-to-sequence tasks
        )



        print(peft_model.print_trainable_parameters())
        peft_trainer.train()
        self.model=peft_model
        
        peft_trainer.save_model("./best-t5-peft-model")


        # predictions = []
        # labels=[]

        
        # with torch.no_grad():
        #   for i in tqdm(train_dataset, desc="Evaluating on Validation Set"):
        #       print(i["source"])
        #       input_text=self.tokenizer.encode(self.PREFIX+i["source"],return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(self.device)
        #       output = self.model.generate(input_ids=input_text, max_length=128, num_beams=5,early_stopping=True).to(self.device)
        #       output_padded = output[:, :128]
        #       padding = torch.full((output_padded.size(0), 128 - output_padded.size(1)),
        #                   fill_value=self.tokenizer.pad_token_id,
        #                   dtype=torch.long).to(self.device)
        #       output_padded = torch.cat([output_padded, padding], dim=1)
        #       predictions.append(output_padded[0].cpu().numpy())
        #       target_text =self.P1.apply_corrections(i["source"], i["corrections"])
        #       label_id=self.tokenizer.encode(target_text,return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(self.device)
        #       labels.append(label_id[0].cpu().numpy())

        # predictions = torch.tensor(predictions).to(self.device)
        # labels = torch.tensor(labels).to(self.device)
        # print(predictions.shape,labels.shape)

        # results = self.evaluate1((predictions, labels))
        # print("Evaluation Results:", results)





    def batch_correct(self, sentences: List[str]) -> List[str]:
        

        corrected_sentences = []

        # Process in batches (optional batching logic can be added later if needed)
        for sentence in tqdm(sentences, desc="Correcting Sentences"):
            input_text = self.PREFIX + sentence
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", 
                                              padding="max_length", truncation=True, 
                                              max_length=128).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        # Decode the output
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_sentences.append(corrected_text)
        return corrected_sentences

    def save(self, path: str):
        """
        Save the model and tokenizer to the specified path.
        """
        import os

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self,path:str, config):
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        MODEL_NAME="t5-base"
        self.tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model=T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)

        from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
        peft_model = PeftModel.from_pretrained(self.model, path)
        self.model=peft_model


    @classmethod
    def load_and_prepare_data(cls, path: str, config: Optional[GECConfig] = None):
        """
        Load the model and tokenizer.

        Args:
            path: Directory containing the model and tokenizer
            config: Optional configuration object

        Returns:
            Loaded GECorrector instance
        """
        P1= M2Parser()
        L=P1.parse_m2_file(path)
        print("Training instance generated", L[0]) ## we have the source and correction till here.

        from sklearn.model_selection import train_test_split
        train_data,val_data=train_test_split(L,test_size=0.1,random_state=42)
        return train_data,val_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEC using BART")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--m2_file", type=str, help="Path to M2 file for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--source_file", type=str, help="Path to source sentences")
    parser.add_argument("--reference_file", type=str, help="Path to reference corrections")
    parser.add_argument("--correct", action="store_true", help="Correct sentences")
    parser.add_argument("--input_file", type=str, help="Path to input sentences")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--model_path", type=str, default="./gec_model_outputs", help="Path to save/load model")
    parser.add_argument("--test_m2_file", type=str, help="Path to M2 file for evaluation")
    args = parser.parse_args()

    config = GECConfig(output_dir=args.model_path)

    if args.train and args.m2_file:
        corrector = GECorrector(config)
        train_dataset, val_dataset = corrector.load_and_prepare_data(args.m2_file)
        corrector.train(train_dataset, val_dataset)
        corrector.save(args.model_path)
    else:
        corrector = GECorrector.load(args.model_path, config)


    if args.evaluate and args.source_file and args.reference_file:
        results = corrector.evaluate(args.source_file, args.reference_file)
        logger.info(f"Evaluation results: {results}")

    if args.correct and args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        corrected_sentences = corrector.batch_correct(sentences)

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for sentence in corrected_sentences:
                    f.write(f"{sentence}\n")
            logger.info(f"Corrected sentences saved to {args.output_file}")
        else:
            for original, corrected in zip(sentences, corrected_sentences):
                logger.info(f"Original: {original}")
                logger.info(f"Corrected: {corrected}")
                logger.info("-" * 50)