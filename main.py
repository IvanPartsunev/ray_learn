import ray
import daft

items = [{"name": str(i), "data": i} for i in range(10000)]
df = daft.from_pylist(items)

result = (df.with_column("data", daft.col("data").apply(lambda x: x ** 2, return_dtype=daft.DataType.int64()))
          .where(df["data"] % 2 == 0)).with_column("data", daft.col("data").apply(lambda x: [x, x ** 3],
                                                                                  return_dtype=daft.DataType.list(
                                                                                    daft.DataType.float64()))).explode(df["data"])
result.show()
