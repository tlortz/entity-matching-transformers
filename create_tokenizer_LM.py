# Databricks notebook source
from pyspark.sql import functions as F
import numpy as np
import os
import tokenizers
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
import torch

# COMMAND ----------

class custom_text_vectorizer():
  
  def __init__(self,temp_dir,tokenizer_dir,transformer_dir,table_name,column_name,sample_frac=1.0,vocab_size=50000,max_input_sequence=512,num_train_epochs=5,use_gpu=False):
    self.temp_dir = temp_dir
    self.tokenizer_dir = tokenizer_dir
    self.transformer_dir = transformer_dir
    self.table_name = table_name
    self.column_name = column_name
    self.sample_frac = sample_frac
    self.vocab_size = vocab_size
    self.max_input_sequence = max_input_sequence
    self.num_train_epochs = num_train_epochs
    self.use_gpu = use_gpu
    
  def _stage_data(self,source_table, source_col, output_dir, file_name='docs.txt'):
    print("staging source data for training...")
    df = spark.read.table(source_table).select(source_col).sample(sample_frac,seed=1)
    text = list(df.toPandas()[source_col].values)
    docs = "[SEP]\n[CLS]".join(text)
    docs = "[CLS]"+docs+"[SEP]"
    file = open(os.path.join(output_dir,file_name),'w')
    file.write(docs)
    file.close()
    print(f"{source_table}.{source_col} content staged to {os.path.join(output_dir,file_name)} for model training")
    
  def _train_tokenizer(self,doc_path,save_path,vocab_size):
    print("training custom text tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=doc_path, vocab_size=vocab_size, min_frequency=2, special_tokens=[
      "[SEP]",
      "[CLS]",
      "<mask>",
    ])
    tokenizer.save_model(save_path)
    print(f"custom text tokenizer saved to {save_path}")
    
  def _train_transformer(self,text_dir,tokenizer_dir,transformer_dir,num_train_epochs,use_gpu):
    print("training custom text transformer model...")
    
    tokenizer = tokenizers.implementations.ByteLevelBPETokenizer(
      os.path.join(tokenizer_dir,'vocab.json'),
      os.path.join(tokenizer_dir,'merges.txt')
    )
    
    tokenizer._tokenizer.post_processor = BertProcessing(
      ("[SEP]", tokenizer.token_to_id("[SEP]")),
      ("[CLS]", tokenizer.token_to_id("[CLS]")),
    )
    
    tokenizer.enable_truncation(max_length=512)
    vocab_size = tokenizer.get_vocab_size() + 4
    
    config = RobertaConfig(
      vocab_size=vocab_size,
      max_position_embeddings=514,
      num_attention_heads=12,
      num_hidden_layers=6,
      type_vocab_size=1,
    )
    
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_length=512, truncation=True)
    
    model = RobertaForMaskedLM(config=config)
    
    dataset = LineByLineTextDataset(
      tokenizer=tokenizer,
      file_path=os.path.join(text_dir,'docs.txt'),
      block_size=128,
    )
    
    data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
      output_dir= transformer_dir,
      overwrite_output_dir=True,
      num_train_epochs=num_train_epochs,
      per_device_train_batch_size=64,
      save_steps=10_000,
      save_total_limit=2,
      learning_rate=1e-4,
      prediction_loss_only=True,
#       place_model_on_device = use_gpu,
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=dataset,
    )
    
    if use_gpu:
      model.to(torch.device('cuda'))
      
#     print(f"Using device {trainer.device}")
    
    trainer.train()
    trainer.save_model(transformer_dir)
    print(f"custom text tokenizer saved to {transformer_dir}")
    
  def fit(self):
    self._stage_data(self.table_name,self.column_name,self.temp_dir)
    self._train_tokenizer(os.path.join(self.temp_dir,'docs.txt'),self.tokenizer_dir,self.vocab_size)
    self._train_transformer(text_dir=self.temp_dir,tokenizer_dir=self.tokenizer_dir,transformer_dir=self.transformer_dir,num_train_epochs=self.num_train_epochs,use_gpu=self.use_gpu)
    return None

# COMMAND ----------

### EXAMPLE USAGE ###

# dbutils.fs.mkdirs('dbfs:/tmp/tim.lortz@databricks.com/drug_review_tokenizer_test')
# dbutils.fs.mkdirs('dbfs:/tmp/tim.lortz@databricks.com/drug_review_transformer_test')
# vectorizer = custom_text_vectorizer(temp_dir="/tmp",tokenizer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_tokenizer_test',\
#                                     transformer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_transformer_test',table_name='tim_lortz_nlp.drug_reviews_bronze_test',\
#                                    column_name='review',use_gpu=True)
# vectorizer.fit()

# COMMAND ----------

# MAGIC %md Potential future changes:
# MAGIC 
# MAGIC - swap out Roberta for Distilbert to accelerate training time and make model smaller
# MAGIC - add parameter to sample from source table-column, in case the entire dataset is too large or highly redundant
# MAGIC - allow all the TrainerConfig params to be customized per run
# MAGIC - allow the (temporary) input text file to be renamed