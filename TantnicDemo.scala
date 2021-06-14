package SPARK_anal2
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

//逻辑回归

object TantnicDemo {
  // 1.建session
  def main(args:Array[String]):Unit = {
    val spark  = SparkSession  //创建spark session
      .builder()
      .appName(this.getClass.getSimpleName)
      .master(master="local")
      .getOrCreate()
    //2.load data 并且add col
    val data = spark.read.csv(path = "C:/Users/Dell/Downloads/泰坦尼克号幸存者代码、数据集和答案集/代码、数据集和答案集/mytrain_notitle.csv").toDF(
      "PassengerId","Survived","Pclass","Name","Sex",
      "Age","SibSp","Parch","Ticket",
      "Fare","Cabin","Embarked")  //csv 不能有中文标题
    //3.求年龄平均值
    val age_mean = data.filter(conditionExpr = "Age is not null")
      .agg(aggExpr = "Age"->"avg").first().getDouble(0)

    //println("平均年龄是："+age_mean)

    spark.udf.register("sexC",{x:String =>
      x match {
        case "male" => 1
        case _=>0
      }
    })
    spark.udf.register("ageC",{x:String=>
      x match{
        case null => age_mean
        case _=>x.toDouble
      }
    })

    val pre_data = data.selectExpr("sexC(Sex) as Sex",
      "ageC(Age) as Age","Pclass","SibSp","Parch","Fare","Survived")  //

    pre_data.show(6)  //dataframe 可以 .show
    val columns = pre_data.columns
    val train_data = pre_data.select(columns.map(x=>col(x).cast(DoubleType)):_*)
    val assembler = new VectorAssembler()
      .setInputCols(Array("Sex","Age","Pclass","SibSp","Parch","Fare"))
      .setOutputCol("features")
    val train_datas = assembler.transform(train_data)
    train_datas.show(10)



    ///建立逻辑回归模型
//    val lor_model = new LogisticRegression()
//      .setFamily("multinomial")
//      .setRegParam(0.3)
//      .setMaxIter(10)
//      .setElasticNetParam(0.8)
//      .setFeaturesCol("features")  //指定特征
//      .setLabelCol("Survived")     //指定标签
//      .fit(train_datas)
    //println("模型情况：迭代次数是"+lor_model.summary.totalIterations) //模型的总体情况 迭代次数
    // lor_model.save(spark.sparkContext,"路径") 需要保存模型 + 路径 spark.sparkContext是环境
    // sameModel = LogisticRegressionModel.load(环境, "路径")
    //println(s"Multinomial coefficients: ${lor_model.coefficientMatrix}")
    //println(s"Multinomial intercepts: ${lor_model.interceptVector}")
    //print(lor_model.coefficientMatrix)
    //print(lor_model.interceptVector)

    val naiveBayes = new NaiveBayes() //朴素贝叶斯模型
      //.setModelType("gaussian")  //默认mti
      .setFeaturesCol("features")
      .setLabelCol("Survived")
      .fit(train_datas)

    val rf = new RandomForestClassifier() //随机森林模型
      .setFeaturesCol("features")
      .setLabelCol("Survived")
      .fit(train_datas)

    println(rf.toString()) //由多个决策树组成

    val fullPredictons = rf.transform(train_datas).cache()
    val predictions = fullPredictons
      .select("prediciton")
      .rdd.map(_.getDouble(0))
    val labels = fullPredictons.select("Survived")
      .rdd.map(_.getDouble(0))
    val UnderROC = new BinaryClassificationMetrics(
      predictions.zip(labels)).areaUnderROC()
    println(UnderROC)

  }
}
