"""Memory budget for the tiled renderer: chosen from the GPU's memory, adjusted while training.

budget_mb (passed to render_tiled) = memory for blending ONE group of tiles. Two more numbers:
  limit  how much memory this process may use: the GPU's size on CUDA; on a Mac, the
         working-set size Metal recommends (CPU and GPU share the RAM, so not all of it)
  used   how much is in use right now

Start (the budget formula):
  budget = min(max_mb, fraction * max(0, limit - used - reserve_mb))
Every `measure_every` steps, measure the step's peak memory and adjust:
  peak above `high` x limit -> budget x 0.7      peak below `low` x limit -> budget x 1.25
On "out of memory": budget x 0.5 and the step is retried.

Measuring the peak:
  CUDA  PyTorch counts it directly (reset, run the step, read the maximum).
  Mac   PyTorch has no peak counter. Its memory pool keeps freed blocks for reuse, so after
        emptying the pool, its size at the end of the step is close to the step's peak.
"""

import torch

MB = 2**20


def limit_and_used_mb(device):
    """(limit, used) in MiB, or (None, None) on the CPU."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return total / MB, (total - free) / MB
    if device.type == "mps":
        return torch.mps.recommended_max_memory() / MB, torch.mps.driver_allocated_memory() / MB
    return None, None


def start_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()  # pool starts empty, so its size after the step ~ its peak


def read_peak_mb(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_reserved(device) / MB
    if device.type == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory() / MB
    return None


def is_out_of_memory(error):
    """CUDA / MPS allocator: "... out of memory ...". Metal itself, on a Mac:
    "... Insufficient Memory (...kIOGPUCommandBufferCallbackErrorOutOfMemory)"."""
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "outofmemory" in message


class MemoryBudget:
    def __init__(
        self,
        device,
        max_mb: float = 4096,
        min_mb: float = 64,
        reserve_mb: float = 1024,
        fraction: float = 0.5,
        measure_every: int = 50,
        high: float = 0.9,
        low: float = 0.7,
    ):
        """
        max_mb, min_mb  the budget always stays between these
        reserve_mb      kept free for everything that is not tile blending: blobs, Adam
                        state, (blob, tile) pairs, sorting, the whole-image buffers
        fraction        share of the remaining free memory the blending may use
        measure_every   measure the peak every this many steps (measuring costs a little)
        high, low       adjust when the peak is above / below these shares of the limit
        """
        self.device = torch.device(device)
        self.max_mb, self.min_mb = max_mb, min_mb
        self.measure_every, self.high, self.low = measure_every, high, low
        self.limit_mb, used_mb = limit_and_used_mb(self.device)
        self.peak_mb = None
        if self.limit_mb is None:  # CPU: nothing to query
            self.budget_mb = min(512, max_mb)
        else:
            free_for_render = fraction * max(0.0, self.limit_mb - used_mb - reserve_mb)
            self.budget_mb = min(max_mb, max(min_mb, free_for_render))
        print(
            f"memory: limit {self.limit_mb or float('nan'):.0f} MiB, "
            f"start budget {self.budget_mb:.0f} MiB per tile group"
        )

    def measuring(self, step):
        return self.limit_mb is not None and step % self.measure_every == 0

    def before_step(self, step):
        if self.measuring(step):
            start_peak(self.device)

    def after_step(self, step):
        """After backward: measure the peak and nudge the budget."""
        if not self.measuring(step):
            return
        self.peak_mb = read_peak_mb(self.device)
        if self.peak_mb > self.high * self.limit_mb:
            self.budget_mb = max(self.min_mb, self.budget_mb * 0.7)
        elif self.peak_mb < self.low * self.limit_mb:
            self.budget_mb = min(self.max_mb, self.budget_mb * 1.25)

    def on_out_of_memory(self):
        """Halve the budget. Returns False if it was already at the minimum (give up)."""
        if self.budget_mb <= self.min_mb:
            return False
        self.budget_mb = max(self.min_mb, self.budget_mb * 0.5)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        print(f"  out of memory: budget -> {self.budget_mb:.0f} MiB, retrying the step")
        return True
