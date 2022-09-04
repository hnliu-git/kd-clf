import os
import transformers

lib_path = '/'.join(transformers.__file__.split('/')[:-1])
cur_path = 'bert/'

common = ['configuration_utils.py', 'modeling_outputs.py']
for name in common:
    lib_file_path = os.path.join(lib_path, name)
    cur_file_path = os.path.join(cur_path, name)
    os.system('rm %s'%lib_file_path)
    os.system('cp %s %s' % (cur_file_path, lib_path))

bert = ['modeling_bert.py', 'configuration_bert.py']
bert_path = os.path.join(lib_path, 'models/bert/')
for name in bert:
    lib_file_path = os.path.join(bert_path, name)
    cur_file_path = os.path.join(cur_path, name)
    os.system('rm %s'%lib_file_path)
    os.system('cp %s %s' % (cur_file_path, bert_path))

roberta = ['modeling_roberta.py']
roberta_path = os.path.join(lib_path, 'models/roberta/')
for name in roberta:
    lib_file_path = os.path.join(roberta_path, name)
    cur_file_path = os.path.join(cur_path, name)
    os.system('rm %s'%lib_file_path)
    os.system('cp %s %s' % (cur_file_path, roberta_path))

