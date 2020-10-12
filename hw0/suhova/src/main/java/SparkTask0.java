import com.github.davidmoten.geo.GeoHash;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.corr;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.first;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.udf;
import static org.apache.spark.sql.functions.variance;

public class SparkTask0 {
    public static SparkSession spark;

    public static void main(String[] args) {
        spark = SparkSession
            .builder()
            .config("spark.master", "local")
            .getOrCreate();

        Dataset<Row> data = spark.read()
            .option("header", "true")
            .option("quote", "\"")
            .option("mode", "DROPMALFORMED")
            .option("escape", "\"")
            .option("inferSchema", "true")
            .csv("hw0/suhova/src/main/resources/AB_NYC_2019.csv");

        //Посчитать медиану, моду, среднее и дисперсию для каждого room_type
        data.withColumn("cnt", count("price").over(Window.partitionBy("room_type")))
            .withColumn("price_mode", first("price").over(Window.orderBy("cnt").partitionBy("room_type")).as("mode"))
            .groupBy("room_type")
            .agg(
                callUDF("percentile_approx", col("price"), lit(0.5)).as("median"),
                first("price_mode").as("mode"),
                avg("price").as("avg"),
                variance("price").as("variance"))
            .show();
/*          +---------------+------+----+------------------+------------------+
            |      room_type|median|mode|               avg|          variance|
            +---------------+------+----+------------------+------------------+
            |    Shared room|  45.0|  40| 70.13298791018998|10365.890682680929|
            |Entire home/apt| 160.0| 225|211.88216032823104| 80852.24645965557|
            |   Private room|  70.0| 149| 89.51396823968689|23907.680804069663|
            +---------------+------+----+------------------+------------------+*/

        // самое дешевое предложение
        data.orderBy("price").show(1);
/*       +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         |      id|                name|host_id|host_name|neighbourhood_group|     neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
         +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         |18750597|Huge Brooklyn Bro...|8993084| Kimberly|           Brooklyn|Bedford-Stuyvesant|40.69023|-73.95428|Private room|    0|             4|                1| 2018-01-06|             0.05|                           4.0|              28|
         +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+ */

        // самое дорогое предложение
        data.orderBy(col("price").desc()).show(1);

/*        +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
          |     id|                name|host_id|host_name|neighbourhood_group|  neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
          +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
          |9528920|Quiet, Clean, Lit...|3906464|      Amy|          Manhattan|Lower East Side|40.71355|-73.98507|Private room| 9999|            99|                6| 2016-01-01|             0.14|                           1.0|              83|
          +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+*/

        //Посчитать корреляцию между ценой и минимальный количеством ночей, кол-вом отзывов
        data.agg(
            corr("price", "minimum_nights").as("corr_nights_price"),
            corr("price", "number_of_reviews").as("corr_review_price"))
            .show();
/*
        +-------------------+--------------------+
        |  corr_nights_price|   corr_review_price|
        +-------------------+--------------------+
        |0.04238800501413225|-0.04806955416645...|
        +-------------------+--------------------+*/

        UserDefinedFunction encodeHash = udf((UDF3<Double, Double, Integer, String>) GeoHash::encodeHash, DataTypes.StringType);
        UserDefinedFunction getLat = udf((UDF1<String, Double>) SparkTask0::getLat, DataTypes.DoubleType);
        UserDefinedFunction getLon = udf((UDF1<String, Double>) SparkTask0::getLon, DataTypes.DoubleType);

        //Нужно найти гео квадрат размером 5км на 5км с самой высокой средней стоимостью жилья
        data.withColumn("hash", encodeHash.apply(col("latitude").cast(DataTypes.DoubleType),
            col("longitude").cast(DataTypes.DoubleType),
            lit(5))
        )
            .withColumn("price", col("price").cast(DataTypes.LongType))
            .groupBy("hash")
            .agg(avg("price").as("avg_price"))
            .orderBy(col("avg_price").desc())
            .limit(1)
            .select(
                getLat.apply(col("hash")).as("latitude"),
                getLon.apply(col("hash")).as("longitude"),
                col("avg_price")
            )
            .show();
/*          +--------------+---------------+---------+
            |      latitude|      longitude|avg_price|
            +--------------+---------------+---------+
            |40.58349609375|-73.71826171875|    350.0|
            +--------------+---------------+---------+*/
    }

    private static Double getLat(String hash) {
        return GeoHash.decodeHash(hash).getLat();
    }

    private static Double getLon(String hash) {
        return GeoHash.decodeHash(hash).getLon();
    }

}