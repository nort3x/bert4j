import com.robrua.nlp.bert.Bert
import org.junit.jupiter.api.Test
import kotlin.test.assertTrue

class LoadTest {
    @Test
    fun `load bert module`() {
        val engine = Bert.Companion.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")
        assertTrue { engine.embedSequence("hello").size == 768 }
        assertTrue { engine.embedSequences("hello", "hi").all { it.size == 768 } }
        assertTrue { engine.embedSequences(listOf("hello", "hi")).all { it.size == 768 } }
        assertTrue { engine.embedSequences(listOf("hello", "hi").iterator()).all { it.size == 768 } }
    }
}