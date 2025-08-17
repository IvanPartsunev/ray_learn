import ray
import subprocess


def map_function(document):
  for word in document.lower().split():
    yield word, 1


@ray.remote
def apply_map(cor, num_parts = 3):
  results = [list() for _ in range(num_parts)]
  for document in cor:
    for result in map_function(document):
      first_letter = result[0].decode("utf-8")[0]
      word_index = ord(first_letter) % num_partitions
      results[word_index].append(result)
  return results


@ray.remote
def apply_reduce(*map_res):
  results = dict()
  for result in map_res:
    for key, value in result:
      results[key] = results.get(key, 0) + value
  return results


zen_of_python = subprocess.check_output(["python", "-c", "import this"])
corpus = zen_of_python.split()

num_partitions = 3
len_corpus = len(corpus)
chunk = len_corpus // num_partitions

partitions = [
  corpus[i * chunk: (i + 1) * chunk] for i in range(num_partitions)
]

map_results = [
  apply_map.options(num_returns=num_partitions).remote(data, num_partitions)
  for data in partitions
]

outputs = []
for i in range(num_partitions):
  outputs.append(
    apply_reduce.remote(*[partition[i] for partition in map_results])
  )

counts = {k: v for output in ray.get(outputs) for k, v in output.items()}

sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

for count in sorted_counts:
  print(f"{count[0].decode('utf-8')}: {count[1]}")