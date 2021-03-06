import sys
from lang_model_lstm import LSTMLanguageModel

def main(fname):
	lm_object = LSTMLanguageModel()
	infile = open(fname)
	first = True
	input_seq = []
	for line in infile:
		if first:
			first = False
			continue
		text_id, text, author = line.split('","')
		if not author.startswith("EAP"): continue
		input_seq.extend(lm_object.text_to_seq(text.lower()))
	input_seq = lm_object.get_input(input_seq)
	print(input_seq.shape)
	lm_object.finish_corpus()
	lm_object.generate_reverse_index()
	lm_object.train_model(input_seq)


if __name__=="__main__":
	main("/Users/ambermadvariya/Documents/585/project/data/train.csv")
