"""Toy LLM serving simulator: static vs continuous batching under a KV-token capacity.
One 'tick' = one decode step for every running request (prefill of a request costs `prefill_ticks`)."""
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    arrival: int
    prompt: int
    out: int
    start: int = -1
    done: int = -1
    produced: int = 0
    first_token: int = -1

    @property
    def kv(self):            # KV tokens currently held
        return self.prompt + self.produced

    @property
    def kv_max(self):
        return self.prompt + self.out


def simulate(reqs, capacity_tokens, max_batch=8, mode="continuous", prefill_ticks=1, reserve_full=True, max_queue=None):
    """mode: 'static' (wait for batch to drain before admitting) or 'continuous' (admit whenever there is room).
    reserve_full: admission reserves prompt+max_out KV tokens (safe, no preemption). Returns (finished, rejected, log)
    where log[t] = (running_ids, kv_used)."""
    reqs = [Req(**vars(r)) for r in reqs]
    pending = sorted(reqs, key=lambda r: (r.arrival, r.rid))
    queue, running, finished, rejected, log = [], [], [], [], []
    t, busy_until = 0, 0
    while pending or queue or running:
        while pending and pending[0].arrival <= t:
            r = pending.pop(0)
            if max_queue is not None and len(queue) >= max_queue:
                rejected.append(r)             # backpressure / admission control: shed load
            else:
                queue.append(r)
        can_admit = (mode == "continuous") or (mode == "static" and not running)
        if can_admit and t >= busy_until:
            used = sum((r.kv_max if reserve_full else r.kv) for r in running)
            while queue and len(running) < max_batch:
                r = queue[0]
                need = r.kv_max if reserve_full else r.prompt
                if r.kv_max > capacity_tokens:           # can never fit
                    rejected.append(queue.pop(0)); continue
                if used + need > capacity_tokens:
                    break
                queue.pop(0); r.start = t; running.append(r); used += need
                busy_until = max(busy_until, t) + prefill_ticks   # prefill stalls the engine
                if mode == "static" and len(running) == max_batch:
                    break
        if t >= busy_until and running:
            for r in running:
                r.produced += 1
                if r.first_token < 0:
                    r.first_token = t
            for r in [r for r in running if r.produced >= r.out]:
                r.done = t + 1; running.remove(r); finished.append(r)
        log.append(([r.rid for r in running], sum(r.kv for r in running)))
        t += 1
        if t > 100000:
            raise RuntimeError("simulation did not terminate")
    return finished, rejected, log


def metrics(finished, log):
    n = len(finished)
    mk = max((r.done for r in finished), default=0)
    return dict(requests=n, makespan=mk, throughput=sum(r.out for r in finished) / max(mk, 1),
                mean_latency=sum(r.done - r.arrival for r in finished) / max(n, 1),
                mean_ttft=sum(r.first_token - r.arrival for r in finished) / max(n, 1),
                peak_kv=max((k for _, k in log), default=0))
