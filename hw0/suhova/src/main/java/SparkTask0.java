import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.concat;
import static org.apache.spark.sql.functions.corr;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.first;
import static org.apache.spark.sql.functions.lit;
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

        // самое дешевое предложение
        data.orderBy("price").show(1);
        // самое дорогое предложение
        data.orderBy(col("price").desc()).show(1);

        //Посчитать корреляцию между ценой и минимальный количеством ночей, кол-вом отзывов
        data.agg(
            corr("price", "minimum_nights").as("corr_nights_price"),
            corr("price", "number_of_reviews").as("corr_review_price"))
            .show();
     /*
     Нужно найти гео квадрат размером 5км на 5км с самой высокой средней стоимостью жилья

     Есть версия с поиском среди гео квадратов, в правом нижнем углу которых есть жильё,
      что явно исключает гору потенциально более дорогих квадратов, но ничего лучше не придумалось :/
      P.S. 5 км = 0.059 * 1 градус долготы = 0.045 * 1 градус 40 широты (вроде бы)
      */
        Dataset<Row> square = data.select(
            col("id"),
            col("latitude").cast(DataTypes.DoubleType),
            col("longitude").cast(DataTypes.DoubleType));
        Dataset<Row> houses = data.select(
            col("price").cast(DataTypes.IntegerType),
            col("latitude").cast(DataTypes.DoubleType).as("x"),
            col("longitude").cast(DataTypes.DoubleType).as("y"));

        square.join(houses, col("x").between(col("latitude"), col("latitude").plus(0.045))
            .and(col("y").between(col("longitude"), col("longitude").plus(0.059))))
            .groupBy("id", "latitude", "longitude")
            .agg(
                avg("price").as("avg")
            )
            .orderBy(col("avg").desc())
            .limit(1)
            .select(
                concat(col("latitude"), lit(":"), col("latitude").plus(0.045)).as("latitude"),
                concat(col("longitude"), lit(":"), col("longitude").plus(0.059)).as("longitude")
            )
            .show();
    }
}