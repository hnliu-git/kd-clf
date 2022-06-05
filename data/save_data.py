"""
Script that generates the train-dev-test data from Hg sst2 and tweet
"""
import datasets
from datasets import load_dataset

sst2 = datasets.load_from_disk('sst2')
print(sst2)

exit()


def split_and_save(data_dict, data, save_path):
    train_val_test = data.train_test_split(0.2, 0.8)
    val_test = train_val_test['test'].train_test_split(0.5, 0.5)

    data_dict['train'] = train_val_test['train']
    data_dict['validation'] = val_test['train']
    data_dict['test'] = val_test['test']

    data_dict.save_to_disk(save_path)


sst2 = load_dataset('glue', 'sst2').rename_column('sentence','text')
tweet = load_dataset('tweet_eval', 'sentiment')

data_sst2 = datasets.concatenate_datasets([
    sst2['train'], sst2['validation']
])

data_tweet = datasets.concatenate_datasets([
    tweet['train'], tweet['validation'], tweet['test']
])


split_and_save(sst2, data_sst2, 'sst2')
split_and_save(tweet, data_tweet, 'tweet')