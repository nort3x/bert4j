package com.robrua.nlp.bert

import java.io.File
import java.io.IOException
import java.net.URISyntaxException
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

/**
 * A port of the BERT FullTokenizer in the [BERT GitHub Repository](https://github.com/google-research/bert).
 *
 * It's used to segment input sequences into the BERT tokens that exist in the model's vocabulary. These tokens are later converted into inputIds for the model.
 *
 * It basically just feeds sequences to the [com.robrua.nlp.bert.BasicTokenizer] then passes those results to the
 * [com.robrua.nlp.bert.WordpieceTokenizer]
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 *
 * @see [The Python tokenization code this is ported from](https://github.com/google-research/bert/blob/master/tokenization.py)
 */
class FullTokenizer @JvmOverloads constructor(vocabularyPath: Path, doLowerCase: Boolean = DEFAULT_DO_LOWER_CASE) :
    Tokenizer() {
    private val basic: BasicTokenizer
    private val vocabulary: Map<String, Int>
    private val wordpiece: WordpieceTokenizer

    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabulary
     * the BERT vocabulary file to use for tokenization
     * @since 1.0.3
     */
    constructor(vocabulary: File) : this(Paths.get(vocabulary.toURI()), DEFAULT_DO_LOWER_CASE) {}

    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabulary
     * the BERT vocabulary file to use for tokenization
     * @param doLowerCase
     * whether to convert sequences to lower case during tokenization
     * @since 1.0.3
     */
    constructor(vocabulary: File, doLowerCase: Boolean) : this(Paths.get(vocabulary.toURI()), doLowerCase) {}
    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabularyPath
     * the path to the BERT vocabulary file to use for tokenization
     * @param doLowerCase
     * whether to convert sequences to lower case during tokenization
     * @since 1.0.3
     */
    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabularyPath
     * the path to the BERT vocabulary file to use for tokenization
     * @since 1.0.3
     */
    init {
        vocabulary = loadVocabulary(vocabularyPath)
        basic = BasicTokenizer(doLowerCase)
        wordpiece = WordpieceTokenizer(vocabulary)
    }

    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabularyResource
     * the resource path to the BERT vocabulary file to use for tokenization
     * @since 1.0.3
     */
    constructor(vocabularyResource: String) : this(toPath(vocabularyResource), DEFAULT_DO_LOWER_CASE) {}

    /**
     * Creates a BERT [com.robrua.nlp.bert.FullTokenizer]
     *
     * @param vocabularyResource
     * the resource path to the BERT vocabulary file to use for tokenization
     * @param doLowerCase
     * whether to convert sequences to lower case during tokenization
     * @since 1.0.3
     */
    constructor(vocabularyResource: String, doLowerCase: Boolean) : this(toPath(vocabularyResource), doLowerCase) {}

    /**
     * Converts BERT sub-tokens into their inputIds
     *
     * @param tokens
     * the tokens to convert
     * @return the inputIds for the tokens
     * @since 1.0.3
     */
    fun convert(tokens: Array<String>): IntArray {
        return tokens.asSequence()
            .map { key -> vocabulary[key]!! }
            .toList()
            .toIntArray()
    }

    override fun tokenize(sequence: String): Array<String> {
        return wordpiece.tokenize(*basic.tokenize(sequence))
            .asSequence()
            .flatMap { values -> values.asSequence() }
            .toList().toTypedArray()
    }

    override fun tokenize(vararg sequences: String): Array<Array<String>> {
        return basic.tokenize(*sequences)
            .asSequence()
            .map { tokens ->
                wordpiece.tokenize(*tokens)
                    .asSequence()
                    .flatMap { values -> values.asSequence() }
                    .toList().toTypedArray()
            }.toList().toTypedArray()
    }

    companion object {
        private const val DEFAULT_DO_LOWER_CASE = false
        private fun loadVocabulary(file: Path): Map<String, Int> {
            val vocabulary: MutableMap<String, Int> = HashMap()
            try {
                Files.newBufferedReader(file, Charset.forName("UTF-8")).use { reader ->
                    var index = 0
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        vocabulary[line!!.trim { it <= ' ' }] = index++
                    }
                }
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
            return vocabulary
        }

        private fun toPath(resource: String): Path {
            return try {
                Paths.get(javaClass::class.java.getResource(resource)?.toURI()!!)
            } catch (e: URISyntaxException) {
                throw RuntimeException(e)
            }
        }
    }
}