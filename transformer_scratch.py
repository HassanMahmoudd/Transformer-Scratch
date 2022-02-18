import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer_Scratch(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        device,
        embedding_size,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        num_heads,
        dropout,
        max_length,
    ):

        super(Transformer_Scratch, self).__init__()

        self.device = device

        self.encoder = Encoder(
            source_vocab_size,
            embedding_size,
            num_encoder_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            target_vocab_size,
            embedding_size,
            num_decoder_layers,
            num_heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )


    def forward(self, source_tokens, target_tokens, source_tokens_mask, target_tokens_mask):

        source_tokens_contextualized_embeddings = self.encoder(
            source_tokens,
            source_tokens_mask
        )

        softmax_logits = self.decoder(
            target_tokens,
            source_tokens_contextualized_embeddings,
            source_tokens_mask,
            target_tokens_mask
        )

        return softmax_logits
