package me.nort3x.bert4j

import java.text.Normalizer
import java.util.*
import kotlin.streams.asSequence

/**
 * A port of the BERT BasicTokenizer in the [BERT GitHub Repository](https://github.com/google-research/bert).
 *
 * The BasicTokenizer is used to segment input sequences into linguistic tokens, which in most cases are words. These tokens can be fed to the
 * [com.robrua.nlp.bert.WordpieceTokenizer] to further segment them into the BERT tokens that are used for input into the model.
 *
 * @author Rob Rua (https://github.com/robrua)
 * @version 1.0.3
 * @since 1.0.3
 *
 * @see [The Python tokenization code this is ported from](https://github.com/google-research/bert/blob/master/tokenization.py)
 */
class BasicTokenizer
/**
 * Creates a BERT [com.robrua.nlp.bert.BasicTokenizer]
 *
 * @param doLowerCase
 * whether to convert sequences to lower case during tokenization
 * @since 1.0.3
 */
    (private val doLowerCase: Boolean) : Tokenizer() {
    private fun stripAndSplit(token: String): String {
        var token = token
        if (doLowerCase) {
            token = stripAccents(token.lowercase(Locale.getDefault()))
        }
        return java.lang.String.join(" ", *splitOnPunctuation(token).toList().toTypedArray())
    }

    override fun tokenize(vararg sequences: String): Array<Array<String>> {
        return setOf(*sequences)
            .map { sequence: String -> cleanText(sequence) }
            .map { sequence: String -> tokenizeChineseCharacters(sequence) }
            .map { sequence: String -> whitespaceTokenize(sequence).toList().toTypedArray() }
            .map { tokens: Array<String> ->
                tokens.asSequence().map { token: String -> stripAndSplit(token) }
                    .flatMap { sequence: String -> whitespaceTokenize(sequence) }
                    .toList().toTypedArray()
            }.toTypedArray()
    }

    override fun tokenize(sequence: String): Array<String> {
        return whitespaceTokenize(tokenizeChineseCharacters(cleanText(sequence))).map { token: String ->
            stripAndSplit(
                token
            )
        }.flatMap {
            whitespaceTokenize(
                it
            )
        }.toList().toTypedArray()
    }

    companion object {
        private val CONTROL_CATEGORIES: Set<Int> = setOf(
            Character.CONTROL.toInt(),
            Character.FORMAT.toInt(),
            Character.PRIVATE_USE.toInt(),
            Character.SURROGATE.toInt(),
            Character.UNASSIGNED.toInt()
        ) // In bert-tensorflow this is any category where the Unicode specification starts with "C"
        private val PUNCTUATION_CATEGORIES: Set<Int> = setOf(
            Character.CONNECTOR_PUNCTUATION.toInt(),
            Character.DASH_PUNCTUATION.toInt(),
            Character.END_PUNCTUATION.toInt(),
            Character.FINAL_QUOTE_PUNCTUATION.toInt(),
            Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
            Character.OTHER_PUNCTUATION.toInt(),
            Character.START_PUNCTUATION.toInt()
        ) // In bert-tensorflow this is any category where the Unicode specification starts with "P"
        private val SAFE_CONTROL_CHARACTERS: Set<Int> = setOf('\t'.code, '\n'.code, '\r'.code)
        private val STRIP_CHARACTERS: Set<Int> = setOf(0, 0xFFFD)
        private val WHITESPACE_CHARACTERS: Set<Int> = setOf(' '.code, '\t'.code, '\n'.code, '\r'.code)
        private fun cleanText(sequence: String): String {
            val builder = StringBuilder()
            sequence.codePoints()
                .filter { codePoint: Int -> !STRIP_CHARACTERS.contains(codePoint) && !isControl(codePoint) }
                .map { codePoint: Int -> if (isWhitespace(codePoint)) ' '.code else codePoint }
                .forEachOrdered { codePoint: Int -> builder.append(Character.toChars(codePoint)) }
            return builder.toString()
        }

        private fun isChineseCharacter(codePoint: Int): Boolean {
            return (codePoint >= 0x4E00 && codePoint <= 0x9FFF || codePoint >= 0x3400 && codePoint <= 0x4DBF || codePoint >= 0x20000 && codePoint <= 0x2A6DF || codePoint >= 0x2A700 && codePoint <= 0x2B73F || codePoint >= 0x2B740) && codePoint <= 0x2B81F || codePoint >= 0x2B820 && codePoint <= 0x2CEAF || codePoint >= 0xF900 && codePoint <= 0xFAFF || codePoint >= 0x2F800 && codePoint <= 0x2FA1F
        }

        private fun isControl(codePoint: Int): Boolean {
            return !SAFE_CONTROL_CHARACTERS.contains(codePoint) && CONTROL_CATEGORIES.contains(
                Character.getType(
                    codePoint
                )
            )
        }

        private fun isPunctuation(codePoint: Int): Boolean {
            return (codePoint >= 33 && codePoint <= 47 || codePoint >= 58) && codePoint <= 64 || codePoint >= 91 && codePoint <= 96 || codePoint >= 123 && codePoint <= 126 || PUNCTUATION_CATEGORIES.contains(
                Character.getType(codePoint)
            )
        }

        private fun isWhitespace(codePoint: Int): Boolean {
            return WHITESPACE_CHARACTERS.contains(codePoint) || Character.SPACE_SEPARATOR.toInt() == Character.getType(
                codePoint
            )
        }

        private fun splitOnPunctuation(token: String): Sequence<String> = sequence {
            val builder = StringBuilder()
            token.codePoints().asSequence().forEach { codePoint ->
                if (isPunctuation(codePoint)) {
                    yield(builder.toString())
                    builder.setLength(0)
                    yield(String(Character.toChars(codePoint)))
                } else {
                    builder.append(Character.toChars(codePoint))
                }
            }
            if (builder.isNotEmpty()) {
                yield(builder.toString())
            }
        }

        private fun stripAccents(token: String): String {
            val builder = StringBuilder()
            Normalizer.normalize(token, Normalizer.Form.NFD).codePoints()
                .filter { codePoint: Int -> Character.NON_SPACING_MARK.toInt() != Character.getType(codePoint) }
                .forEachOrdered { codePoint: Int -> builder.append(Character.toChars(codePoint)) }
            return builder.toString()
        }

        private fun tokenizeChineseCharacters(sequence: String): String {
            val builder = StringBuilder()
            sequence.codePoints().forEachOrdered { codePoint: Int ->
                if (isChineseCharacter(codePoint)) {
                    builder.append(' ')
                    builder.append(Character.toChars(codePoint))
                    builder.append(' ')
                } else {
                    builder.append(Character.toChars(codePoint))
                }
            }
            return builder.toString()
        }
    }
}