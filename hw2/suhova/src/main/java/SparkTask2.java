import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.from_json;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

public class SparkTask2 {
    public static SparkSession spark;

    public static void main(String[] args) {
        spark = SparkSession.builder().config("spark.master", "local").getOrCreate();
        Dataset<Row> socketDF = spark
            .readStream()
            .format("socket")
            .option("host", "localhost")
            .option("port", 9999)
            .load();
        StructType schema = new StructType()
            .add("id", IntegerType)
            .add("text", DataTypes.StringType);

        Dataset<Row> recievedJson = socketDF.withColumn("json", from_json(col("value"), schema))
                .select("json.*")
                .filter(col("text").isNotNull())
                .filter(col("id").isNotNull())
                .select("id", "text");

        PipelineModel model = PipelineModel.read().load("hw1/suhova/pipeline/");
        
        try {
            model.transform(recievedJson)
                .select(col("id"), col("target").cast(IntegerType))
                .repartition(1)
                .writeStream()
                .outputMode("append")
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("path", "path/")
                .option("checkpointLocation", "checkpointLocation/")
                .start()
                .awaitTermination();
        } catch (StreamingQueryException e) {
            e.printStackTrace();
        }
    }
}
