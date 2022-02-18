import torch
import spacy
import de_core_news_sm
from torchtext.data.metrics import bleu_score
import sys
from torchtext.data.utils import get_tokenizer

def remove_empty_lines(input_path, output_path):

    outfile = open(output_path, "w", encoding="UTF-8")
    for line in open(input_path, "r", encoding="UTF-8"):
        if not line.isspace():
            outfile.write(line)

    outfile.close()


def convert_data_to_unique_pairs(english_sentences, german_sentences, output_path_english, output_path_german):
    unique_lines = set()
    outfile_english = open(output_path_english, "w", encoding="UTF-8")
    outfile_german = open(output_path_german, "w", encoding="UTF-8")
    for english_line, german_line in zip(english_sentences,german_sentences):
        if english_line + german_line not in unique_lines:
            outfile_english.write(english_line)
            outfile_german.write(german_line)
            unique_lines.add(english_line + german_line)
    outfile_english.close()
    outfile_german.close()

def convert_data_to_unique(input_path, output_path):
    lines_unique = set()
    outfile = open(output_path, "w", encoding="UTF-8")
    for line in open(input_path, "r", encoding="UTF-8"):
        if line not in lines_unique:
            outfile.write(line)
            lines_unique.add(line)
    outfile.close()

def remove_xml_lines(input_path, output_path):
    outfile = open(output_path, "w", encoding="UTF-8")
    for line in open(input_path, "r", encoding="UTF-8"):
        if not line.startswith("<"):
            outfile.write(line)
    outfile.close()


def read_sentences(input_path):
    sentences = []
    for line in open(input_path, "r", encoding="UTF-8"):
        sentences.append(line)
    return sentences

def get_count_sentences(input_path):
    count = 0
    for line in open(input_path, "r", encoding="UTF-8"):
        count += 1
    return count


def split_data(input_path, num_valid, num_test, output_train_path, output_valid_path, output_test_path):
    outfile_train = open(output_train_path, "w", encoding="UTF-8")
    outfile_valid = open(output_valid_path, "w", encoding="UTF-8")
    outfile_test = open(output_test_path, "w", encoding="UTF-8")

    count = 0
    for line in open(input_path, "r", encoding="UTF-8"):
        if count >= (num_valid + num_test):
            outfile_train.write(line)
        elif count >= num_valid:
            outfile_valid.write(line)
        else:
            outfile_test.write(line)
        count += 1

    outfile_train.close()
    outfile_valid.close()
    outfile_test.close()


def model_inference(model, sentence, german, english, device, max_length, src_pad_idx, is_sentence_tokenized):

    tokenize_german = get_tokenizer('spacy', language='de_core_news_sm')

    if is_sentence_tokenized:
        sentence_tokens = [token for token in sentence]
    else:
        sentence_tokens = [token for token in tokenize_german(sentence)]

    sentence_tokens.insert(0, german.init_token)
    sentence_tokens.append(german.eos_token)

    source_sentence_text_to_indices = [german.vocab.stoi[token] for token in sentence_tokens]

    source_sentence_tensor = torch.LongTensor(source_sentence_text_to_indices).unsqueeze(1)

    prediction = [english.vocab.stoi["<sos>"]]

    for i in range(max_length):
        target_sentence_tensor = torch.LongTensor(prediction).unsqueeze(1)

        with torch.no_grad():

            source_tokens = source_sentence_tensor.to(device)
            target_tokens = target_sentence_tensor.to(device)

            source_tokens_transposed = torch.transpose(source_tokens, 0, 1)
            target_tokens_transposed = torch.transpose(target_tokens, 0, 1)

            source_tokens_mask = make_src_mask(source_tokens_transposed, src_pad_idx).to(device)
            target_tokens_mask = make_trg_mask(target_tokens_transposed).to(device)

            softmax_logits = model(source_tokens_transposed, target_tokens_transposed, source_tokens_mask, target_tokens_mask)

        softmax_logits_transposed = torch.transpose(softmax_logits, 0, 1)

        highest_softmax_logits = softmax_logits_transposed.argmax(2)[-1, :].item()
        prediction.append(highest_softmax_logits)

        if highest_softmax_logits == english.vocab.stoi["<eos>"]:
            break

    prediction_sentence = [english.vocab.itos[token_idx] for token_idx in prediction]

    return prediction_sentence


def calculate_bleu_score(data, model, german, english, device, max_length, src_pad_idx):
    targets = []
    predictions = []

    for sample in data:

        source_sentence = sample.src
        target_sentence = sample.trg
        prediction = model_inference(model, source_sentence, german, english, device, max_length, src_pad_idx, True)
        prediction = prediction[1:-1]

        targets.append(target_sentence)
        predictions.append(prediction)

    score = bleu_score(predictions, targets)
    return score


def make_src_mask(source_tokens, src_pad_idx):
    source_tokens_mask = (source_tokens != src_pad_idx).to(torch.int)
    source_tokens_mask = source_tokens_mask.unsqueeze(1).unsqueeze(2)

    return source_tokens_mask

def make_trg_mask(target_tokens):
    target_sentences_length, target_tokens_length = target_tokens.shape
    target_tokens_mask = torch.ones((target_tokens_length, target_tokens_length))
    target_tokens_trainglular_mask = torch.tril(target_tokens_mask)

    target_tokens_mask = target_tokens_trainglular_mask.expand(target_sentences_length, 1, target_tokens_length, target_tokens_length)

    return target_tokens_mask

