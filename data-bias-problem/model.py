from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .arguments import args


model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
