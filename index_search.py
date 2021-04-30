# Databricks notebook source
import numpy as np
import pandas as pd
import time
import faiss
from transformers import pipeline, RobertaTokenizerFast

# COMMAND ----------

class low_latency_matcher():
  
  def __init__(self, source_table, tokenizer_dir, transformer_dir, index_path, key_col=None, display_cols=None, max_sequence_length=512, use_gpu=False):
    self._key_col = key_col
    self._use_gpu = use_gpu
    self._df = self._load_data(source_table,key_col,display_cols)
    self._tokenizer = self._load_tokenizer(tokenizer_dir,max_sequence_length)
    self._model = self._load_model(transformer_dir,self._tokenizer)
    self._index = self._load_index(index_path)
    
  def _load_data(self,table,key_col,display_cols):
    print(f"staging source data from table {table} for matching...")
    if key_col is not None:
      select_cols = display_cols.copy()
      select_cols.insert(0,key_col)
      df = spark.table(table).select(select_cols)
      return df.toPandas().set_index(keys=key_col,inplace=False,drop=False)
    else:
      df = spark.table(table)#.select(content_col)
      return df.toPandas().reset_index()

  def _load_tokenizer(self,tokenizer_dir,max_sequence_length):
    print(f"loading pretrained tokenizer from {tokenizer_dir}...")
    return RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_length=max_sequence_length, truncation=True)

  def _load_model(self,transformer_dir,tokenizer):
    print(f"loading pretrained transformer model from {transformer_dir}...")
    device = -1
    if self._use_gpu is True:
      device = 0
    return pipeline("feature-extraction", model=transformer_dir, tokenizer=tokenizer, device=int(device))
  
  def _load_index(self,index_path):
    return faiss.read_index(index_path)

  def match(self,query,k=5):
    t=time.time()
    input_text = "[CLS]"+query+"[SEP]"
    raw_embeds = np.array(self._model(input_text))
    query_vector = np.mean(raw_embeds[:,1:-1,:],axis=1).astype("float32").reshape(1,-1)
    faiss.normalize_L2(query_vector)
    top_k = self._index.search(query_vector, k)
#     print(f"top {k} index IDs: {top_k[1].tolist()}")
#     print(top_k)
    print('total time: {}'.format(time.time()-t))
    results = self._df.loc[[_ for _ in top_k[1].tolist()[0]]].copy()
    results['similarity'] = top_k[0].tolist()[0]
    return results

# COMMAND ----------

# matcher = low_latency_matcher(source_table='tim_lortz_nlp.drug_reviews_bronze_test',\
#                               tokenizer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_tokenizer_test',\
#                              transformer_dir='/dbfs/tmp/tim.lortz@databricks.com/drug_review_transformer_test',\
#                              index_path='/dbfs/tmp/tim.lortz@databricks.com/drug_review_index_test/tim_lortz_nlp.drug_reviews_bronze_test.review.index',\
#                              use_gpu=True)

# COMMAND ----------

# r = matcher.match('My depression and anxiety got worse after I started taking it.',k=20)
# display(r)