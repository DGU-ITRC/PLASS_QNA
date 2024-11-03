import argparse
import os
import torch
import random
import numpy as np
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

def init_args(context, question):
    if context is None or question is None:
        context = "Stephen Silvagni (born 31 May 1967) is a former Australian rules footballer for the Carlton Football Club." 
        question = "What was the name of Stephen Silvagni's team?"
    args = {
        'context': context,
        'question': question,
        'seed': 42,
        'save_dir': 'save/baseline-01',
    }
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def predict(context, question):
    args = init_args(context, question)
    set_seed(args['seed'])
    checkpoint_path = os.path.join(args['save_dir'], 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    args['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    context = args['context']
    question = args['question']
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, )
    start_index = outputs.start_logits.argmax()
    end_index = outputs.end_logits.argmax()
    start_idx = int(start_index.numpy())
    end_idx = int(end_index.numpy())   
    predict_tokens = inputs.input_ids[0,start_index:end_index+1]
    predict_answer = tokenizer.decode(predict_tokens)
    result = {'context': context, 'question': question, 'start_idx': start_idx, 'end_idx': end_idx, 'answer': predict_answer}
    print(result)
    return result


if __name__ == '__main__':
    context = "Stephen Silvagni (born 31 May 1967) is a former Australian rules footballer for the Carlton Football Club." 
    question = "What was the name of Stephen Silvagni's team?"
    predict(context, question)
