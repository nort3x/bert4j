package me.nort3x.bert4j

/**
 * A port of the BERT WordpieceTokenizer in the [BERT GitHub Repository](https://github.com/google-research/bert).
 *
 * The WordpieceTokenizer processes tokens from the [com.robrua.nlp.bert.BasicTokenizer] into sub-tokens - parts of words that compose BERT's vocabulary.
 * These tokens can then be converted into the inputIds that the BERT model accepts.
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 *
 * @see [The Python tokenization code this is ported from](https://github.com/google-research/bert/blob/master/tokenization.py)
 */
class WordpieceTokenizer : Tokenizer {
    private val maxCharactersPerWord: Int
    private val unknownToken: String
    private val vocabulary: Map<String, Int>

    /**
     * Creates a BERT [com.robrua.nlp.bert.WordpieceTokenizer]
     *
     * @param vocabulary
     * a mapping from sub-tokens in the BERT vocabulary to their inputIds
     * @since 1.0.3
     */
    constructor(vocabulary: Map<String, Int>) {
        this.vocabulary = vocabulary
        unknownToken = DEFAULT_UNKNOWN_TOKEN
        maxCharactersPerWord = DEFAULT_MAX_CHARACTERS_PER_WORD
    }

    /**
     * Creates a BERT [com.robrua.nlp.bert.WordpieceTokenizer]
     *
     * @param vocabulary
     * a mapping from sub-tokens in the BERT vocabulary to their inputIds
     * @param unknownToken
     * the sub-token to use when an unrecognized or too-long token is encountered
     * @param maxCharactersPerToken
     * the maximum number of characters allowed in a token to be sub-tokenized
     * @since 1.0.3
     */
    constructor(vocabulary: Map<String, Int>, unknownToken: String, maxCharactersPerToken: Int) {
        this.vocabulary = vocabulary
        this.unknownToken = unknownToken
        maxCharactersPerWord = maxCharactersPerToken
    }

    private fun splitToken(token: String): Sequence<String> {
        val characters = token.toCharArray()
        if (characters.size > maxCharactersPerWord) {
            return sequenceOf(unknownToken)
        }
        return sequence {
            var start = 0
            var end: Int
            while (start < characters.size) {
                end = characters.size
                var found = false
                while (start < end) {
                    val substring = (if (start > 0) "##" else "") + String(characters, start, end - start)
                    if (vocabulary.containsKey(substring)) {
                        yield(substring)
                        start = end
                        found = true
                        break
                    }
                    end--
                }
                if (!found) {
                    yield(unknownToken)
                    break
                }
                start = end
            }

        }

    }

    override fun tokenize(sequence: String): Array<String> {
        return whitespaceTokenize(sequence)
            .flatMap { token: String -> splitToken(token) }
            .toList().toTypedArray()

    }

    override fun tokenize(vararg sequences: String): Array<Array<String>> {
        return sequences.asSequence()
            .map { sequence: String -> whitespaceTokenize(sequence).toList().toTypedArray() }
            .map { tokens: Array<String> ->
                tokens.asSequence()
                    .flatMap { token: String -> splitToken(token) }
                    .toList().toTypedArray()
            }.toList().toTypedArray()
    }

    companion object {
        private const val DEFAULT_MAX_CHARACTERS_PER_WORD = 200
        private const val DEFAULT_UNKNOWN_TOKEN = "[UNK]"
    }
}