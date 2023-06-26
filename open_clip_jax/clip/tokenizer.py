"""
Byte-pair encoding tokenization in TensorFlow. This is copied from
https://github.com/keras-team/keras-nlp/ but contains a few small modifications
since CLIP's tokenizer is different than that of GPT-2.
"""

# Modifications are marked by MOD: Description of modification.

# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gzip
import os

import tensorflow as tf
import tensorflow_text as tf_text


# As python and TF handles special spaces differently, we need to
# manually handle special spaces during string split.
SPECIAL_WHITESPACES = r"\x{a0}\x{2009}\x{202f}\x{3000}"

# String splitting regex pattern.
SPLIT_PATTERN_1 = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    + r"|[\s{special_spaces}]+[\n\r\t\f६{special_spaces}]|\p{L}+|"
    + r"[\p{N}]|[^\s\p{L}\p{N}{special_spaces}]+"
)
SPLIT_PATTERN_1 = SPLIT_PATTERN_1.replace(
    "{special_spaces}", SPECIAL_WHITESPACES
)


SPLIT_PATTERN_2 = rf"""[\s६{SPECIAL_WHITESPACES}]$"""



def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # removes mapping an int to a whitespace character
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    bs = [n.to_bytes(1, "little") for n in bs]
    return bs, cs  # int to string mapping


def remove_strings_from_inputs(tensor, string_to_remove):
    """Remove certain strings from input tensor."""
    non_empty_mask = tensor != string_to_remove
    flatten_indexes = tf.where(non_empty_mask)
    flatten_result = tf.gather_nd(tensor, flatten_indexes)
    row_lengths = tf.reduce_sum(tf.cast(non_empty_mask, tf.int64), axis=1)
    result = tf.RaggedTensor.from_row_lengths(
        values=flatten_result,
        row_lengths=row_lengths,
    )
    return result


def split_strings_for_bpe(inputs):
    # We need to recreate the exact behavior of token presplitting in the
    # original gpt2 tokenizer which uses a lookahead. As re2 does not
    # support lookahead match, we are using an alternative insert a special
    # token "६" before leading space of non-space characters and after the
    # trailing space, e.g., " keras" will be "६ keras".
    inputs = tf.strings.regex_replace(
        inputs, rf"( )([^\s{SPECIAL_WHITESPACES}])", r"६\1\2"
    )
    inputs = tf.strings.regex_replace(
        inputs, rf"(\s{SPECIAL_WHITESPACES})$", r"\1६"
    )
    raw_tokens = tf_text.regex_split(inputs, SPLIT_PATTERN_1, SPLIT_PATTERN_1)
    # Second pass splits out the last whilespace char or "६".
    raw_tokens = tf_text.regex_split(
        raw_tokens, SPLIT_PATTERN_2, SPLIT_PATTERN_2
    )
    if raw_tokens.shape.rank > 2:
        raw_tokens = raw_tokens.merge_dims(1, 2)
    # MOD: Spaces are not tokenized.
    raw_tokens = remove_strings_from_inputs(raw_tokens, " ")
    return remove_strings_from_inputs(raw_tokens, "६")


class BytePairTokenizerCache(tf.Module):
    """Cache that stores the encoded result of seen tokens.

    The cache key is string tensor or python strings, and the value is split
    tokens joined by whitespace. For example, "dragonfly" => "dragon fly"

    Examples:
    ```
    cache = BytePairTokenizerCache()
    cache.insert(["butterfly", "dragonfly"], ["but ter fly", "dragon fly"])
    cache.lookup(["butterfly"])
    ```
    """
    def __init__(self):
        # `tf.lookup.experimental.MutableHashTable` does not support string to
        # string mapping. So we first convert to string to an integer key, and
        # use the integer key to find the value.
        self.factors = tf.pow(256, tf.range(0, 8, dtype=tf.int64))
        self.id2value = tf.lookup.experimental.MutableHashTable(
            tf.int64, tf.string, ""
        )

    def _get_key(self, keys):
        """Get the hash key for given inputs."""
        # `tf.fingerprint` converts token to a array of uint8 of length 8, we
        # need to convert it to a uint64.
        return tf.squeeze(
            tf.matmul(
                tf.cast(tf.fingerprint(keys), dtype=tf.int64),
                self.factors[:, tf.newaxis],
            ),
            -1,
        )

    def lookup(self, keys):
        """Look up the encoded outputs of given tokens."""
        ids = self._get_key(keys)
        result = self.id2value.lookup(ids)
        # Ensure output shape for graph mode.
        result.set_shape([None])
        return result

    def insert(self, keys, values):
        """Insert token <=> encoded outputs pairs."""
        self.id2value.insert(self._get_key(keys), values)


def create_static_hashtable(keys, values, default):
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(keys),
            tf.convert_to_tensor(values),
        ),
        default_value=default,
    )


# MOD: BytePairTokenizer does not inherit a tokenizer class or Keras layer.
class BytePairTokenizer:
    """Bype-pair encoding tokenizer layer.

    If input is a batch of strings (rank > 0):
    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged. If `sequence_length` is set, the layer
    will output a dense `tf.Tensor` where all inputs have been padded or
    truncated to `sequence_length`.
    If input is a scalar string (rank == 0):
    By default, the layer will output a dense `tf.Tensor` with static shape
    `[None]`. If `sequence_length` is set, the output will be
    a dense `tf.Tensor` of shape `[sequence_length]`.

    Args:
        bpe_path: Path to BPE vocab file.
        sequence_length: int, defaults to None. If set, the output will be
            padded or truncated to the `sequence_length`.
        add_prefix_space: bool, defaults to False. Whether or not to add an
            initial space to the input. This tokenizer is whitespace aware,
            and will tokenize a word with a leading space differently. Adding
            a prefix space to the first word will cause it to be tokenized
            equivalently to all subsequent words in the sequence.
    """

    # MOD: Some arguments have been removed.
    def __init__(
        self,
        bpe_path=default_bpe(),
        sequence_length=None,
        add_prefix_space=False,
    ) -> None:
        byte_list, unicode_list = bytes_to_unicode()

        # MOD: This bit has been copied from https://github.com/openai/CLIP
        # and extracts the vocabulary and merge rules.
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = unicode_list
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        special_tokens = ['<start_of_text>', '<end_of_text>']
        vocab.extend(special_tokens)
        vocab = dict(zip(vocab, range(len(vocab))))
        self.vocabulary = dict(zip(vocab, range(len(vocab))))
        self.merges = dict(zip(merges, range(len(merges))))
        self.merges = [" ".join(m) for m in self.merges.keys()]

        self.sequence_length = sequence_length
        self.add_prefix_space = add_prefix_space

        # Create byte <=> unicode mapping. This is useful for handling
        # whitespace tokens.
        self.byte2unicode = create_static_hashtable(
            byte_list, unicode_list, default=""
        )
        self.unicode2byte = create_static_hashtable(
            unicode_list, byte_list, default=""
        )

        self.cache = BytePairTokenizerCache()

        # Create mapping between string tokens to int ids, and vice versa.
        byte_pairs = [x[0] for x in self.vocabulary.items()]
        byte_pair_encoding_indices = [x[1] for x in self.vocabulary.items()]
        self.token_to_id_map = create_static_hashtable(
            byte_pairs,
            byte_pair_encoding_indices,
            default=-1,
        )
        self.id_to_token_map = create_static_hashtable(
            byte_pair_encoding_indices,
            byte_pairs,
            default="",
        )

        # Create ranking of merge rules, this is the same as order of merge
        # pairs in `self.merges`.
        self.merge_ranks_lookup_default = len(self.merges) + 1
        self.merge_ranks = create_static_hashtable(
            self.merges,
            list(range(len(self.merges))),
            default=self.merge_ranks_lookup_default,
        )


    @tf.function
    def _bpe_merge_one_step(self, words, mask):
        """Perform one step of byte-pair merge."""
        # Get all word pairs.
        first, second = words[:, :-1], words[:, 1:]

        # Mask empty.
        non_empty_mask = second.nested_row_lengths()[0] != 0
        mask = mask & non_empty_mask
        if not tf.reduce_any(mask):
            return [words, mask]
        non_empty_indices = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        filterd_first = tf.ragged.boolean_mask(first, mask)
        filtered_second = tf.ragged.boolean_mask(second, mask)

        # Get byte pair ranking in merge rules.
        pairs = tf.strings.join([filterd_first, filtered_second], separator=" ")
        pair_rank = self.merge_ranks.lookup(pairs)

        # Get BPE pair ranks.
        min_pair_rank = tf.reduce_min(pair_rank, axis=1)
        pair_found_mask = min_pair_rank != self.merge_ranks_lookup_default

        # Tokens that cannot be further merged are marked as finished.
        mask = tf.tensor_scatter_nd_update(
            mask, tf.expand_dims(non_empty_indices, axis=1), pair_found_mask
        )
        if not tf.math.reduce_any(mask):
            return [words, mask]

        masked_pair_rank = tf.ragged.boolean_mask(pair_rank, pair_found_mask)
        min_pair_rank_indices = tf.math.argmin(
            masked_pair_rank.to_tensor(self.merge_ranks_lookup_default), axis=1
        )

        # Get words and pairs to process.
        unfinished_words = tf.ragged.boolean_mask(words, mask)

        pair_left = tf.gather(
            unfinished_words, min_pair_rank_indices, batch_dims=1
        )
        pair_right = tf.gather(
            unfinished_words, min_pair_rank_indices + 1, batch_dims=1
        )

        merged_pairs = tf.strings.join([pair_left, pair_right])
        empty_strs = tf.fill(tf.shape(merged_pairs), "")

        unfinished_word_indices = tf.cast(
            tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask), dtype=tf.int64
        )
        merged_pair_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis],
            ],
            axis=1,
        )
        empty_string_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis] + 1,
            ],
            axis=1,
        )

        tensor_words = words.to_tensor(default_value="")
        tensor_words = tf.tensor_scatter_nd_update(
            tensor_words,
            merged_pair_indices,
            merged_pairs,
        )

        words = tf.tensor_scatter_nd_update(
            tensor_words,
            empty_string_indices,
            empty_strs,
        )
        # Remove empty strings.
        words = remove_strings_from_inputs(words, "")
        return [words, mask]

    def _bpe_merge(self, inputs):
        """Perform byte-pair merge for each word in the inputs."""

        # MOD: Word boundaries are identified by appending '</w>' to each word.
        inputs = tf.concat([inputs[:, :-1], inputs[:, -1:]+'</w>'], axis=1)
        num_words = tf.shape(inputs)[0]

        # Merge bytes.
        def loop_condition(_, mask):
            return tf.math.reduce_any(mask)

        initial_mask = tf.fill((num_words,), True)
        merged_words, _ = tf.while_loop(
            loop_condition,
            self._bpe_merge_one_step,
            loop_vars=[
                inputs,
                initial_mask,
            ],
            shape_invariants=[
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
            ],
        )
        return merged_words

    def __call__(self, inputs, context_len=77):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        # MOD: Whitespace cleaning and lowercase casting.
        inputs = tf.strings.strip(inputs)
        inputs = tf.strings.regex_replace(inputs, r'\s+', ' ')
        inputs = tf.strings.lower(inputs)

        if self.add_prefix_space:
            inputs = tf.strings.join([" ", inputs])

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        raw_tokens = split_strings_for_bpe(inputs)
        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""

        has_unseen_words = tf.math.reduce_any(
            (cache_lookup == "") & (flat_tokens != "")
        )

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )

        tokens = tf.strings.split(tokenized_words, sep=" ")
        tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

        # MOD: The tokens are bracketed with start- and end-of-stream tokens.
        sos = tf.constant([self.vocabulary['<start_of_text>']])
        eos = tf.constant([self.vocabulary['<end_of_text>']])
        shape = (tf.shape(tokens)[0], 1)
        tokens = tf.concat(
            [tf.broadcast_to(sos, shape), tokens, tf.broadcast_to(eos, shape)],
             axis=1,
            )

        # Convert to a dense output if context_len is set.
        if context_len:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = context_len
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input in scalar
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[context_len])

        # MOD: If a single data point is passed, a batch axis is added.
        if tf.rank(tokens) == 1:
            tokens = tf.expand_dims(tokens, axis=0)

        return tokens

    def _transform_bytes(self, tokens):
        """Map token bytes to unicode using `byte2unicode`."""
        split_bytes = tf.strings.bytes_split(tokens)
        split_unicode = self.byte2unicode.lookup(split_bytes)
        return split_unicode

    def _bpe_merge_and_update_cache(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._transform_bytes(tokens)
        tokenized_words = self._bpe_merge(words)

        # For each word, join all its token by a whitespace,
        # e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, tokenized_words)


tokenize = BytePairTokenizer()
