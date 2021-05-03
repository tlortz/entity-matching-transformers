# Databricks notebook source
import math
import numpy as np
import os
import pandas as pd
from transformers import pipeline, RobertaTokenizerFast
import faiss
import torch

# COMMAND ----------

class text_vectorizer_indexer():
  
  def __init__(self, table, content_col, transformer_dir, tokenizer_dir, index_dir, key_col=None, index_name=None, use_gpu=False, index_on_key=False, save_embeddings=False, embeddings_dir=None, save_embeddings_format='delta', max_sequence_length=512,batch_size=4):
    self.table = table
    self.content_col = content_col
    self.transformer_dir = transformer_dir
    self.tokenizer_dir = tokenizer_dir
    self.index_dir = index_dir
    self.key_col = key_col
    self.index_name = index_name
    self.use_gpu = use_gpu
    self.index_on_key = index_on_key
    self.save_embeddings = save_embeddings
    self.embeddings_dir = embeddings_dir
    self.save_embeddings_format = save_embeddings_format
    self.max_sequence_length = max_sequence_length
    self.batch_size = batch_size
    self.raw_df = self._load_raw_data(table,content_col,key_col)#.iloc[:2000]
    self.tokenizer = self._load_tokenizer(tokenizer_dir,max_sequence_length)
    self.transformer = self._load_transformer(transformer_dir,self.tokenizer)
    print(self.raw_df.head())
    
  def _load_raw_data(self,table,content_col,key_col):
    print(f"staging source data from table {table} for training...")
    if key_col is not None:
      df = spark.table(table).select(key_col,content_col)
      return df.toPandas()
    else:
      df = spark.table(table).select(content_col)
      return df.toPandas().reset_index()
    
  def _load_tokenizer(self,tokenizer_dir,max_sequence_length):
    print(f"loading pretrained tokenizer from {tokenizer_dir}...")
    return RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_length=max_sequence_length, truncation=True)
  
  def _load_transformer(self,transformer_dir,tokenizer):
    print(f"loading pretrained transformer from {transformer_dir}...")
    device = -1
    if self.use_gpu is True:
      device = 0
    return pipeline("feature-extraction", model=transformer_dir, tokenizer=tokenizer, device=int(device))
  
  def _create_embeddings(self):
    num_records = self.raw_df.shape[0]
    print(f"creating embeddings vector for {num_records} records...")
    raw_text = self.raw_df[self.content_col].tolist()
    num_batches = math.ceil(num_records/self.batch_size)
    results = []
    for i in range(num_batches):
      start_range = i*self.batch_size
      end_range = min((i+1)*self.batch_size,num_records)
      print(f"processing records {start_range+1} through {end_range}")
      input_text = ["[CLS]"+t[:1200]+"[SEP]" for t in raw_text[start_range:end_range]]
#       input_text = ["[CLS]"+t+"[SEP]" for t in raw_text[start_range:end_range]]
      raw_embeds = self.transformer(input_text)
      doc_embeds = [np.mean(e[1:-1],axis=0) for e in raw_embeds]
      results = results + doc_embeds
    results = np.array(results).astype("float32")
    print(f"created embeddings with dimension {results.shape}")
    return results
  
  def _save_embeddings(self,embeddings):
    print(f"appending embeddings column to source table in {self.embeddings_dir}...")
    file_format = self.save_embeddings_format
    df = self.raw_df.copy()
    df[self.content_col + '_embeddings'] = embeddings.tolist()
    if file_format == 'delta':
      spark.createDataFrame(df).write.format('delta').save(self.embeddings_dir)
    if file_format == 'parquet':
      spark.createDataFrame(df).write.format('parquet').save(self.embeddings_dir)
    if file_format == 'csv':
      df.to_csv(self.embeddings_dir)
  
  def _create_index(self,embeddings):
    #expects embeddings to be of type numpy.ndarray
    print(f"populating index...")
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    if self.index_on_key:
      ids = np.array(self.raw_df[self.key_col]).flatten().astype("int")
    else:
      ids = np.array(range(self.raw_df.shape[0]))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(embeddings_normalized, ids)
    return index
  
  def _save_index(self,index):
    print(f"saving populated index in {self.index_dir}...")
    if self.index_name is None:
      name = self.table + '.' + self.content_col + '.index' 
    else: 
      name = self.index_name
    path = os.path.join( self.index_dir, name)
    faiss.write_index(index,path)
  
  def transform(self):
    embeddings = self._create_embeddings()
    index = self._create_index(embeddings)
    if self.save_embeddings:
      self._save_embeddings(embeddings)
    self._save_index(index)
    return embeddings, index

# COMMAND ----------

#%fs mkdirs dbfs:/tmp/tim.lortz@databricks.com/drug_review_index_test

# COMMAND ----------

#%fs mkdirs dbfs:/tmp/tim.lortz@databricks.com/drug_review_embeddings_test

# COMMAND ----------

# tvi = text_vectorizer_indexer(table='tim_lortz_nlp.drug_reviews_bronze_test',content_col='review',key_col='a',\
#                               transformer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_transformer_test',\
#                               tokenizer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_tokenizer_test',\
#                               index_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_index_test',\
#                               embeddings_dir='dbfs:/tmp/tim.lortz@databricks.com/drug_review_embeddings_test',\
#                               use_gpu=True,save_embeddings=True,index_on_key=False)

# COMMAND ----------

#embeddings, index = tvi.transform()

# COMMAND ----------

#%fs ls dbfs:/tmp/tim.lortz@databricks.com/