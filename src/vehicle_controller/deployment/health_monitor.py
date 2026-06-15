"""Runtime health statistics."""

from dataclasses import dataclass


@dataclass
class HealthMonitor:
    inference_timeout_ms: float
    inference_count: int = 0
    timeout_count: int = 0
    maximum_inference_ms: float = 0.0

    def observe_inference(self, elapsed_ms: float) -> bool:
        self.inference_count += 1
        self.maximum_inference_ms = max(self.maximum_inference_ms, elapsed_ms)
        timed_out = elapsed_ms > self.inference_timeout_ms
        self.timeout_count += int(timed_out)
        return not timed_out

