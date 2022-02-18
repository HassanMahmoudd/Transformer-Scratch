import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):

        super(MultiHeadAttention, self).__init__()

        self.embedding_size = embedding_size

        self.num_heads = num_heads

        self.head_embedding_size = embedding_size // num_heads

        self.queries = nn.Linear(self.head_embedding_size, self.head_embedding_size, False)
        self.keys = nn.Linear(self.head_embedding_size, self.head_embedding_size, False)
        self.values = nn.Linear(self.head_embedding_size, self.head_embedding_size, False)

        self.fcnn = nn.Linear(embedding_size, embedding_size)

    def forward(self, query, keys, values, mask):

        num_source_sentences = query.shape[0]

        query_sequence_length = query.shape[1]
        key_sequence_length = keys.shape[1]
        value_sequence_length = values.shape[1]

        query = query.reshape(num_source_sentences, query_sequence_length, self.num_heads, self.head_embedding_size)
        keys = keys.reshape(num_source_sentences, key_sequence_length, self.num_heads, self.head_embedding_size)
        values = values.reshape(num_source_sentences, value_sequence_length, self.num_heads, self.head_embedding_size)

        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        queries_key_multiplication = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        queries_key_multiplication_masked = queries_key_multiplication.masked_fill(mask == 0, float("-inf"))

        queries_key_multiplication_masked_attention = torch.softmax(queries_key_multiplication_masked / (self.head_embedding_size ** (1 / 2)), dim = 3)

        query_contextualized_embeddings = torch.einsum("nhqv,nvhd->nqhd", [queries_key_multiplication_masked_attention, values]).reshape(
            num_source_sentences, query_sequence_length, self.embedding_size
        )

        query_contextualized_embeddings = self.fcnn(query_contextualized_embeddings)

        return query_contextualized_embeddings


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, forward_expansion):
        super(MultiHeadAttentionBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_size, num_heads)

        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):

        query_contextualized_embeddings = self.multi_head_attention(query, key, value, mask)

        query_contextualized_embeddings_normalized = self.dropout(self.layer_norm_1(query_contextualized_embeddings + query))

        query_contextualized_embeddings_normalized_feed_forward = self.feed_forward(query_contextualized_embeddings_normalized)

        query_contextualized_embeddings_normalized_2 = self.layer_norm_2(query_contextualized_embeddings_normalized_feed_forward + query_contextualized_embeddings_normalized)

        query_contextualized_embeddings_normalized_feed_forward_normalized = self.dropout(query_contextualized_embeddings_normalized_2)

        return query_contextualized_embeddings_normalized_feed_forward_normalized

