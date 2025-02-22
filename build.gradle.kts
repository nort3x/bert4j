import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.21"
    `maven-publish`
}
val groupVar = "me.nort3x"
val versionVar = "0.0.1-SNAPSHOT"

group = groupVar
version = versionVar
java.sourceCompatibility = JavaVersion.VERSION_17
description =
    "A Dead Simple BERT API (https://github.com/google-research/bert) refactored and bumped - forked from: https://github.com/robrua/easy-bert"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.tensorflow:tensorflow-core-api:0.4.1")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.13.3")
    testImplementation("com.robrua.nlp.models:easy-bert-uncased-L-12-H-768-A-12:1.0.0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}
publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = groupVar
            artifactId = "bert4j"
            version = versionVar

            from(components["java"])
        }
    }
}