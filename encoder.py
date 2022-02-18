import torch
import torch.nn as nn

from multi_head_attention_block import MultiHeadAttentionBlock

class Encoder(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        embedding_size,
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):

        super(Encoder, self).__init__()

        self.device = device

        encoder_layers = []

        for i in range(num_layers):

            multi_head_attention_encoder_encoder = MultiHeadAttentionBlock(
                    embedding_size,
                    num_heads,
                    dropout,
                    forward_expansion
            )

            encoder_layers.append(multi_head_attention_encoder_encoder)

        self.encoder_layers = nn.ModuleList(encoder_layers)

        self.dropout = nn.Dropout(dropout)

        self.word_embedding = nn.Embedding(source_vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(max_length, embedding_size)

    def forward(self, source_tokens, source_tokens_mask):

        num_source_sentences, source_tokens_sequence_length = source_tokens.shape

        absolute_positions = torch.arange(0, source_tokens_sequence_length)

        absolute_positions = absolute_positions.expand(num_source_sentences, source_tokens_sequence_length).to(self.device)

        source_tokens_embeddings = self.dropout((self.word_embedding(source_tokens) + self.position_embedding(absolute_positions)))

        source_tokens_contextualized_embeddings = source_tokens_embeddings

        for encoder_layer in self.encoder_layers:

            source_tokens_contextualized_embeddings = encoder_layer(source_tokens_contextualized_embeddings, source_tokens_embeddings, source_tokens_embeddings, source_tokens_mask)

        return source_tokens_contextualized_embeddings