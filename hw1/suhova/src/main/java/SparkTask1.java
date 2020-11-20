import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.when;

public class SparkTask1 {
    public static SparkSession spark;

    public static void main(String[] args) {
        spark = SparkSession.builder().config("spark.master", "local").getOrCreate();
        Dataset<Row> train = spark.read().option("header", "true").option("inferSchema", "true")
            .csv("hw1/suhova/src/main/resources/train.csv")
            .filter(col("text").isNotNull())
            .filter(col("target").isNotNull())
            .select("id", "text", "target")
            .withColumnRenamed("target", "label");

        Dataset<Row> test = spark.read().option("header", "true").option("inferSchema", "true")
            .csv("hw1/suhova/src/main/resources/test.csv")
            .filter(col("text").isNotNull())
            .filter(col("id").isNotNull())
            .select("id", "text");

        Dataset<Row> sample = spark.read().option("header", "true").option("inferSchema", "true")
            .csv("hw1/suhova/src/main/resources/sample_submission.csv")
            .select("id");

        RegexTokenizer regexTokenizer = new RegexTokenizer()
            .setInputCol("text")
            .setOutputCol("words")
            .setPattern("\\W");

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
            .setInputCol("words")
            .setOutputCol("removed");

        Stemmer stemmer = new Stemmer()
            .setInputCol("removed")
            .setOutputCol("stemmed");

        HashingTF hashingTF = new HashingTF()
            .setInputCol("stemmed")
            .setNumFeatures(3000)
            .setOutputCol("rawFeatures");

        IDF idf = new IDF()
            .setInputCol("rawFeatures")
            .setOutputCol("features");

        StringIndexer labelIndexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel");

        GBTClassifier gbt = new GBTClassifier()
            .setLabelCol("indexedLabel")
            .setFeaturesCol("features")
            .setPredictionCol("target")
            .setMaxIter(30);

        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{
                regexTokenizer, stopWordsRemover, stemmer, hashingTF, idf, labelIndexer, gbt
            });

        Dataset<Row> result = pipeline.fit(train).transform(test)
            .select(col("id"), col("target").cast(DataTypes.IntegerType));

        result = result.join(sample, sample.col("id").equalTo(result.col("id")), "right")
            .select(sample.col("id"),
                when(result.col("id").isNull(), lit(0))
                    .otherwise(col("target"))
                    .as("target")
            );

        result.write().option("header", "true").option("inferSchema", "true")
            .csv("hw1/suhova/src/main/resources/result.csv");
    }
}