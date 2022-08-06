## bert4j

originally fork from [easy-bert](https://github.com/robrua/easy-bert/)

aiming to:

* maintaining apis to match the latest version of `tensorflow`
* adding `kotlin` api support
* improving performance managed parts

### how to get it?

#### part1: adding library api

for gradle:

```kotlin
// .kts
repositories {
    maven { url = URI("https://jitpack.io") }
}


dependencies {
    implementation("com.github.nort3x:bert4j:-SNAPSHOT")
    // you need to also add platform but probably specific one - otherwise your artifact going to be huge!
    // read more on: https://github.com/tensorflow/java#using-maven-artifacts
    implementation("org.tensorflow:tensorflow-core-platform:0.4.1")
    // you may also want these to actually build something
    implementation("org.tensorflow:tensorflow-core-api:0.4.1")
    implementation("org.tensorflow:tensorflow-framework:0.4.1")
}
```

for maven:

```xml

<!--please read guide in the gradle section above-->

<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependency>
    <groupId>com.github.nort3x</groupId>
    <artifactId>bert4j</artifactId>
    <version>-SNAPSHOT</version>
</dependency>

```

#### part2 (optional) : adding models as dependency

visit [easy-bert](https://github.com/robrua/easy-bert#pre-generated-maven-central-models)
developer robrua provided necessities as dependency

for instance:
```kotlin
//gradle:
dependencies {
    // uncased English 12-layers 768-embeddingSize 12-head 110M-parameters
    // check at: https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
    implementation("com.robrua.nlp.models:easy-bert-uncased-L-12-H-768-A-12:1.0.0")
}
```

#### part3: use
visit [easy-bert](https://github.com/robrua/easy-bert) for in depth document

example - using above configurations:
```kotlin
class BertVectorizer {
    lateinit var bert: Bert


    fun construct() {
        logger().info("loading bert module")
        val loadingTime = measureTimeMillis { bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")}
        logger().info("loaded bert model in $loadingTime ms")
    }

    fun vectorize(input: Sequence<String>, chunkSize: Int = 50): Sequence<FloatArray> =
        input.chunked(chunkSize)
            .map { bert.embedSequences(*it.toTypedArray()) }
            .flatMap { it.asSequence() }


    fun vectorize(input: String): FloatArray =
        bert.embedSequence(input)

}

```