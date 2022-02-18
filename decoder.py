import torch
import torch.nn as nn

from multi_head_attention_block import MultiHeadAttentionBlock, MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self,
        embedding_size,
        num_heads,
        forward_expansion,
        dropout
    ):

        super(DecoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(embedding_size)

        self.multi_head_attention_decoder_decoder = MultiHeadAttention(embedding_size, num_heads)

        self.multi_head_attention_encoder_decoder = MultiHeadAttentionBlock(
            embedding_size,
            num_heads,
            dropout,
            forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, target_tokens_embeddings, source_tokens_contextualized_embeddings, source_tokens_mask, target_tokens_mask):

        multi_head_attention_decoder_decoder = self.multi_head_attention_decoder_decoder(target_tokens_embeddings, target_tokens_embeddings, target_tokens_embeddings, target_tokens_mask)

        multi_head_attention_decoder_decoder_normalized = self.layer_norm(multi_head_attention_decoder_decoder + target_tokens_embeddings)

        target_tokens_contextualized_embeddings = self.dropout(multi_head_attention_decoder_decoder_normalized)

        multi_head_attention_encoder_decoder = self.multi_head_attention_encoder_decoder(target_tokens_contextualized_embeddings, source_tokens_contextualized_embeddings, source_tokens_contextualized_embeddings, source_tokens_mask)

        return multi_head_attention_encoder_decoder


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embedding_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()

        self.device = device

        decoder_layers = []

        for i in range(num_layers):
            decoder_block = DecoderBlock(
                embedding_size,
                num_heads,
                forward_expansion,
                dropout
            )

            decoder_layers.append(decoder_block)

        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.fcnn_softmax_logits = nn.Linear(embedding_size, target_vocab_size)

        self.dropout = nn.Dropout(dropout)

        self.word_embedding = nn.Embedding(target_vocab_size, embedding_size)

        self.position_embedding = nn.Embedding(max_length, embedding_size)

    def forward(self,target_tokens, source_tokens_contextualized_embeddings, source_tokens_mask, target_tokens_mask):

        num_target_sentences, target_tokens_sequence_length = target_tokens.shape

        absolute_positions = torch.arange(0, target_tokens_sequence_length)

        absolute_positions = absolute_positions.expand(num_target_sentences, target_tokens_sequence_length).to(self.device)

        target_tokens_embeddings = self.dropout((self.word_embedding(target_tokens) + self.position_embedding(absolute_positions)))

        target_tokens_contextualized_embeddings = target_tokens_embeddings

        for decoder_layer in self.decoder_layers:

            target_tokens_contextualized_embeddings = decoder_layer(target_tokens_contextualized_embeddings, source_tokens_contextualized_embeddings, source_tokens_mask, target_tokens_mask)

        softmax_logits = self.fcnn_softmax_logits(target_tokens_contextualized_embeddings)

        return softmax_logits