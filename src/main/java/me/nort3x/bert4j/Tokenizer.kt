package me.nort3x.bert4j

/**
 * A tokenizer that converts text sequences into tokens or sub-tokens for BERT to use
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 */
abstract class Tokenizer {
    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     * the sequences to tokenize
     * @return the tokens in the sequences, in the order the [java.lang.Iterable] provided them
     * @since 1.0.3
     */
    fun tokenize(sequences: Iterable<String>): Array<Array<String>> {
        val list: List<String> = sequences.toList()
        return tokenize(*list.toTypedArray())
    }

    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     * the sequences to tokenize
     * @return the tokens in the sequences, in the order the [java.util.Iterator] provided them
     * @since 1.0.3
     */
    fun tokenize(sequences: Iterator<String>): Array<Array<String>> {
        val list: List<String> = sequences.asSequence().toList()
        return tokenize(*list.toTypedArray())
    }

    /**
     * Tokenizes a single sequence
     *
     * @param sequence
     * the sequence to tokenize
     * @return the tokens in the sequence
     * @since 1.0.3
     */
    abstract fun tokenize(sequence: String): Array<String>

    /**
     * Tokenizes a multiple sequences
     *
     * @param sequences
     * the sequences to tokenize
     * @return the tokens in the sequences, in the order they were provided
     * @since 1.0.3
     */
    abstract fun tokenize(vararg sequences: String): Array<Array<String>>

    companion object {
        /**
         * Splits a sequence into tokens based on whitespace
         *
         * @param sequence
         * the sequence to split
         * @return a stream of the tokens from the stream that were separated by whitespace
         * @since 1.0.3
         */
        @JvmStatic
        protected fun whitespaceTokenize(sequence: String): Sequence<String> {
            return sequence
                .trim { it <= ' ' }
                .split("\\s+".toRegex()).dropLastWhile { it.isEmpty() }
                .asSequence()
        }
    }
}