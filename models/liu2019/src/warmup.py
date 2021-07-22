# Pre-cache the BERT model so that it does not need to be downloaded
# every time the Dockerfile runs. We point to the cache_dir that the code
# uses by default.
from pytorch_transformers import BertModel, BertTokenizer

# The BertTokenizer is loaded with a specific cache and the default cache
BertModel.from_pretrained("bert-base-uncased", cache_dir="PreSumm/temp")
BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="PreSumm/temp")
BertTokenizer.from_pretrained("bert-base-uncased")
