import ray
import time

ray.init(address="auto")

database = [
  "Learning", "Ray",
  "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]

db_object_ref = ray.put(database)


def retrieve(item, db):
  time.sleep(item / 10.)
  return item, db[item]


def print_runtime(input_data, start_time):
  print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
  print(*input_data, sep="\n")


@ray.remote
class DataTracker:

  def __init__(self):
    self._count = 0

  def increment(self):
    self._count += 1
  def counts(self):
    return self._count


@ray.remote
def retrieve_task(item, tracker: DataTracker, db):
  time.sleep(item / 10.)
  tracker.increment.remote()
  return item, db[item]


@ray.remote
def follow_up_task(retrieve_results, db):
  original_item, _ = retrieve_results
  follow_up_result = retrieve(original_item + 1, db)
  return retrieve_results, follow_up_result


tracker = DataTracker.remote()

object_references = [
  retrieve_task.remote(item, tracker, db_object_ref)
  for item in range(8)
]

data = ray.get(object_references)
print(data)
print(ray.get(tracker.counts.remote()))

# start = time.time()
# object_references = [retrieve_task.remote(item, db_object_ref) for item in [0, 2, 4, 6]]
# follow_up_refs = [follow_up_task.remote(ref, database) for ref in object_references]
# result = [print(data) for data in ray.get(follow_up_refs)]
# all_data = []

# while len(object_references) > 0:
#   finished, object_references = ray.wait(
#     object_references, num_returns=2, timeout=7.0
#   )
#   data = ray.get(finished)
#   print_runtime(data, start)
#   all_data.extend(data)
