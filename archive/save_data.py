"""
Script that generates the train-dev-test data from Hg sst2 and tweet
"""
import datasets
from datasets import load_dataset


sst2 = load_dataset('glue', 'sst2').rename_column('sentence','text')
sst2.save_to_disk('sst2')

tweet = load_dataset('tweet_eval', 'sentiment')
tweet_test = tweet['test'].train_test_split(0.1, 0.9)
tweet_train = datasets.concatenate_datasets([
    tweet['train'], tweet_test['train']
])
tweet['train'] = tweet_train
tweet['test'] = tweet_test['test']
tweet.save_to_disk('tweet')
