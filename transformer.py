import torch
import torch.nn as nn
from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Field, BucketIterator
from torchtext.datasets import IWSLT2016

import os
import numpy as np
import spacy
import de_core_news_sm
import en_core_web_sm
from torchtext.data.utils import get_tokenizer

from helper import model_inference, calculate_bleu_score, make_src_mask, make_trg_mask
from transformer_scratch import Transformer_Scratch
from helper import convert_data_to_unique, convert_data_to_unique_pairs, get_count_sentences, split_data, read_sentences, remove_empty_lines, remove_xml_lines

data_path = "data/WMT13/"
data_file_name = "data"

english_sentences = read_sentences(data_path + data_file_name + ".en")
german_sentences = read_sentences(data_path + data_file_name + ".de")

print("Example sample sentence from English corpus:", english_sentences[10000])
print("Example parallel sample sentence from German corpus:", german_sentences[10000])

count_english_sentences = get_count_sentences(data_path + data_file_name + ".en")
count_german_sentences = get_count_sentences(data_path + data_file_name + ".de")

convert_data_to_unique_pairs(english_sentences, german_sentences, data_path + data_file_name + "_unique.en", data_path + data_file_name + "_unique.de")

count_english_sentences_unique = get_count_sentences(data_path + data_file_name + "_unique.en")
count_german_sentences_unique = get_count_sentences(data_path + data_file_name + "_unique.de")

num_validation = 1000
num_test = 1000

split_data(data_path + data_file_name + "_unique.en", num_validation, num_test, data_path + "train.en", data_path + "valid.en", data_path + "test.en")
split_data(data_path + data_file_name + "_unique.de", num_validation, num_test, data_path + "train.de", data_path + "valid.de", data_path + "test.de")

english_sentences_train = read_sentences(data_path + "train.en")
german_sentences_train = read_sentences(data_path + "train.de")

print("Example sample sentence from English training corpus:", english_sentences_train[10000])
print("Example parallel sample sentence from German training corpus:", german_sentences_train[10000])



count_english_train_sentences = get_count_sentences(data_path + "train.en")
count_german_train_sentences = get_count_sentences(data_path + "train.de")

tokenize_english = get_tokenizer('spacy', language = 'en_core_web_sm')
tokenize_german = get_tokenizer('spacy', language = 'de_core_news_sm')

german = Field(tokenize = tokenize_german,
               lower = False,
               init_token = "<sos>",
               eos_token = "<eos>"
)
english = Field(tokenize = tokenize_english,
                lower = False,
                init_token = "<sos>",
                eos_token = "<eos>"
)

train_data, valid_data, test_data = TranslationDataset.splits(
    exts = (".de", ".en"),
    fields = (german, english),
    path = data_path,
    train = 'train',
    validation = 'valid',
    test = 'test'
)

print("Example sample sentence tokens from German training corpus:", train_data.examples[10000].src)
print("Example parallel sample sentence tokens from English training corpus:", train_data.examples[10000].trg)

german.build_vocab(train_data, min_freq = 1)
english.build_vocab(train_data, min_freq = 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5
learning_rate = 5e-4
batch_size = 64

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)

embedding_size = 512
num_heads = 8

num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10

max_length_source = max([len(item.src) for item in train_data.examples])
max_length_target = max([len(item.trg) for item in train_data.examples])
max_length = max(max_length_source, max_length_target) + 2
max_length = 100

forward_expansion = 4

src_pad_idx = english.vocab.stoi["<pad>"]
trg_pad_idx = german.vocab.stoi["<pad>"]

sort_within_batch = True
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    sort_within_batch = sort_within_batch,
    sort_key = lambda data: len(data.src),
    device = device
)

model = Transformer_Scratch(
    src_vocab_size,
    trg_vocab_size,
    device,
    embedding_size,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    num_heads,
    dropout,
    max_length
).to(device)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

src_pad_idx = english.vocab.stoi["<pad>"]

cross_entropy = nn.CrossEntropyLoss(ignore_index = src_pad_idx)

example_sentence = "In letzter Zeit allerdings ist dies schwieriger denn je, ist doch der Goldpreis im letzten Jahrzehnt um über 300 Prozent angestiegen."

model.eval()
example_sentence_translated = model_inference(
    model,
    example_sentence,
    german,
    english,
    device,
    max_length,
    src_pad_idx,
    False
)
model.train()

print("Example sentence from test set: ", example_sentence)
print("Example sentence from test set translated: ", " ".join(example_sentence_translated))


print("***** Start training *****")

model_output_path = "epochs_checkpoints/"
lowest_validation_loss = float("inf")
best_epoch_idx = -1
step = 0

for epoch in range(num_epochs):
    print("Epoch", epoch + 1, "from", num_epochs)
    losses = []

    for batch_idx, batch in enumerate(train_iterator):

        source_tokens = batch.src.to(device)
        target_tokens = batch.trg.to(device)

        source_tokens_trimmed = source_tokens[:max_length, :]
        target_tokens_trimmed = target_tokens[:max_length, :]

        source_tokens_transposed = torch.transpose(source_tokens_trimmed, 0, 1)
        target_tokens_transposed = torch.transpose(target_tokens_trimmed[:-1, :], 0, 1)

        source_tokens_mask = make_src_mask(source_tokens_transposed, src_pad_idx).to(device)
        target_tokens_mask = make_trg_mask(target_tokens_transposed).to(device)

        softmax_logits = model(source_tokens_transposed, target_tokens_transposed, source_tokens_mask, target_tokens_mask)

        softmax_logits_transposed = torch.transpose(softmax_logits, 0, 1)
        softmax_logits_flattened = softmax_logits_transposed.reshape(-1, softmax_logits_transposed.shape[2])

        target_tokens_trimmed = target_tokens_trimmed[1:].reshape(-1)

        optimizer.zero_grad()

        cross_entropy_loss = cross_entropy(softmax_logits_flattened, target_tokens_trimmed)
        losses.append(cross_entropy_loss.item())

        cross_entropy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

        optimizer.step()
        step += 1



    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

    print("Epoch {} average training loss: {}".format(epoch + 1, mean_loss))

    valid_losses = []
    model.eval()
    for batch_idx, batch in enumerate(valid_iterator):

        source_tokens = batch.src.to(device)
        target_tokens = batch.trg.to(device)

        source_tokens_trimmed = source_tokens[:max_length, :]
        target_tokens_trimmed = target_tokens[:max_length, :]

        source_tokens_transposed = torch.transpose(source_tokens_trimmed, 0, 1)
        target_tokens_transposed = torch.transpose(target_tokens_trimmed[:-1, :], 0, 1)

        source_tokens_mask = make_src_mask(source_tokens_transposed, src_pad_idx).to(device)
        target_tokens_mask = make_trg_mask(target_tokens_transposed).to(device)

        softmax_logits = model(source_tokens_transposed, target_tokens_transposed, source_tokens_mask, target_tokens_mask)

        softmax_logits_transposed = torch.transpose(softmax_logits, 0, 1)
        softmax_logits_flattened = softmax_logits_transposed.reshape(-1, softmax_logits_transposed.shape[2])

        target_tokens_trimmed = target_tokens_trimmed[1:].reshape(-1)

        valid_loss = cross_entropy(softmax_logits_flattened, target_tokens_trimmed)
        valid_losses.append(valid_loss.item())

        step += 1


    model.train()
    valid_mean_loss = sum(valid_losses) / len(valid_losses)

    print("Epoch {} average validation loss: {}".format(epoch + 1, valid_mean_loss))

    lowest_validation_loss_list = [lowest_validation_loss, valid_mean_loss]
    best_epoch_idx_list = [best_epoch_idx, epoch + 1]

    lowest_validation_loss = lowest_validation_loss_list[np.argmin(lowest_validation_loss_list)]
    best_epoch_idx = best_epoch_idx_list[np.argmin(lowest_validation_loss_list)]

    model.eval()
    example_sentence_translated = model_inference(
        model,
        example_sentence,
        german,
        english,
        device,
        max_length,
        src_pad_idx,
        False
    )
    model.train()

    print("Example sentence from test set: ", example_sentence)
    print("Example sentence from test set translated: ", " ".join(example_sentence_translated))


    print("***** Saving Model Checkpoint *****")
    epoch_output_folder_path = os.path.join(model_output_path, "epoch_{}".format(epoch + 1))

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, epoch_output_folder_path + ".pt")



print("***** Training Ended *****")
print("Lowest Validation Loss is at Epoch:", best_epoch_idx, "with loss of:", lowest_validation_loss)


print("***** Loading Best Model Checkpoint *****")

best_epoch_output_folder_path = os.path.join(model_output_path, "epoch_{}".format(best_epoch_idx))

checkpoint = torch.load(best_epoch_output_folder_path + ".pt")
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])

print("***** Start Evaluation *****")

score = calculate_bleu_score(test_data, model, german, english, device, max_length, src_pad_idx)
print("Test set Bleu score is:", score * 100)

print("***** Evaluation Ended *****")
