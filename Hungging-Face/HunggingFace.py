from transformers import pipeline

# SENTIMENTAL ANALYSIS
sentiment_classifier = pipeline('sentiment-analysis')
sentimentResult = sentiment_classifier("I'm so excited to be learning about large language models")
print('--- sentimentResult: ', sentimentResult, '\n\n')

# NER
ner = pipeline('ner', model='dslim/bert-base-NER')
nerResult = ner("Her name is Anna and she works in New York City for Morgan Stanley")
print('--- nerResult: ', nerResult, '\n\n')

# ZERO CLASSIFIER
zeroshot_classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
zeroResult = zeroshot_classifier(sequence_to_classify, candidate_labels)
print('--- zeroResult: ', zeroResult, '\n\n')

# PRE TRAINED TOKENIZERS - MODEL 1
from transformers import AutoTokenizer
model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)
sentence = "I'm so excited to be learning about large language models"
input_ids = tokenizer(sentence)
print('--- input_ids: ',input_ids, '\n\n')

tokens = tokenizer.tokenize(sentence)
print('--- tokens: ',tokens, '\n\n')

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids, '\n\n')

decoded_ids = tokenizer.decode(token_ids)
print(decoded_ids, '\n\n')

print(tokenizer.decode(101))
print(tokenizer.decode(102), '\n\n')

# PRE TRAINED TOKENIZERS - MODEL 2
model2 = "xlnet-base-cased"
tokenizer2 = AutoTokenizer.from_pretrained(model2)
input_ids2 = tokenizer2(sentence)
print(input_ids2, '\n\n')

tokens2 = tokenizer2.tokenize(sentence)
print(tokens2, '\n\n')

token_ids2 = tokenizer2.convert_tokens_to_ids(tokens2)
print(token_ids2, '\n\n')


# HUNGGINGFACE - Pytorch/Tensorflow
from transformers import AutoModelForSequenceClassification
import torch

tokenizer3 = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
input_ids_pt = tokenizer3(sentence, return_tensors ="pt")
print(input_ids_pt, '\n\n')

model3 = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
with torch.no_grad():
    logits = model3(**input_ids_pt).logits

predicted_class_id = logits.argmax().item()
model3.config.id2label[predicted_class_id]

# SAVING AND OADING MODELS
model_directory = "my_saved_models"
tokenizer3.save_pretrained(model_directory)
my_tokenizer = AutoTokenizer.from_pretrained(model_directory)
my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)