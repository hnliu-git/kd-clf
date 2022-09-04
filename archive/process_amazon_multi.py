from datasets import load_dataset


dataset = load_dataset("amazon_reviews_multi")


# Drop unnecessary columns
dataset = dataset.remove_columns(['review_id', 'product_id', 'reviewer_id', 'review_body', 'language', 'product_category'])


# Map stars to label
def stars2label(example):
    dict_s2l = {1:0, 2:0, 3:1, 4:2, 5:2}
    example['stars'] = dict_s2l[example['stars']]
    return example


dataset = dataset.map(stars2label, num_proc=4).rename_column('stars', 'label').rename_column('review_title', 'text')
dataset.save_to_disk('data/amazon_multi')
