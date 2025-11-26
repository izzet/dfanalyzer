from distributed.diagnostics.plugin import SchedulerPlugin
from time import sleep
from tqdm.auto import tqdm


class Counter(SchedulerPlugin):

    def __init__(self):
        self.tasks_fingerprints = {}

    async def start(self, scheduler):
        self.scheduler = scheduler

    def transition(self, key, start, finish, *args, **kwargs):
        ts = self.scheduler.tasks[key]
        if finish in ['memory', 'released']:
            if ts.prefix_key not in self.tasks_fingerprints:
                self.tasks_fingerprints[ts.prefix_key] = {
                    'sum_dur': 0,
                    'n_tasks': 0
                }
            self.tasks_fingerprints[ts.prefix_key] = {
                'sum_dur': self.tasks_fingerprints[ts.prefix_key]['sum_dur'] + self.scheduler.get_task_duration(ts),
                'n_tasks': self.tasks_fingerprints[ts.prefix_key]['n_tasks'] + 1
            }


class ProgressBarPlugin(SchedulerPlugin):
    def __init__(self):
        self.tqdm_map = {}
        self.ts_count = {}
        self.ts_done = {}

    def start(self, scheduler):
        self.scheduler = scheduler

    def initialize(self, futures):
        self.tqdm_map = {}
        dask_graph = futures.__dask_graph__()
        self.ts_count = {}
        self.ts_done = {}
        for key, value in dask_graph.layers.items():
            node = key.split("-")[0]
            if node not in self.ts_count:
                self.ts_count[node] = 0
            self.ts_count[node] = self.ts_count[node] + 1
        for key, count in self.ts_count.items():
            if key not in self.tqdm_map:
                self.tqdm_map[key] = tqdm(total=self.ts_count[key], desc=f"{key}")
                self.ts_done[key] = 0

    def transition(self, key, start, finish, *args, **kwargs):
        ts = self.scheduler.tasks[key]
        if finish in ["memory", "released"]:
            self.tqdm_map[ts.prefix_key].update(1)
            self.ts_done[ts.prefix_key] = ts.prefix_key + 1

    def finalize(self):
        for key, count in self.ts_count.items():
            self.tqdm_map[key].update(self.ts_count[key] - self.ts_done[key])
            sleep(0.1)
            self.tqdm_map[key].close()
        self.tqdm_map = {}
        self.ts_count = {}
        self.ts_done = {}

    def restart(self, scheduler):
        self.tqdm_map = {}
