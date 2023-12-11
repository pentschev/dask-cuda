import time
from multiprocessing.process import BaseProcess
from typing import Literal, Type, Union

import distributed
from distributed import Nanny, Worker


class MockWorker(Worker):
    """Mock Worker class preventing NVML from getting used by SystemMonitor.

    By preventing the Worker from initializing NVML in the SystemMonitor, we can
    mock test multiple devices in `CUDA_VISIBLE_DEVICES` behavior with single-GPU
    machines.
    """

    def __init__(self, *args, **kwargs):
        distributed.diagnostics.nvml.device_get_count = MockWorker.device_get_count
        self._device_get_count = distributed.diagnostics.nvml.device_get_count
        super().__init__(*args, **kwargs)

    def __del__(self):
        distributed.diagnostics.nvml.device_get_count = self._device_get_count

    @staticmethod
    def device_get_count():
        return 0


class IncreasedCloseTimeoutNanny(Nanny):
    """Increase `Nanny`'s close timeout.

    The internal close timeout mechanism of `Nanny` recomputes the time left to kill
    the `Worker` process based on elapsed time of the close task, which may leave
    very little time for the subprocess to shutdown cleanly, which may cause tests
    to fail when the system is under higher load. This class increases the default
    close timeout of 5.0 seconds that `Nanny` sets by default, which can be overriden
    via Distributed's public API.

    This class can be used with the `worker_class` argument of `LocalCluster` or
    `LocalCUDACluster` to provide a much higher default of 30.0 seconds.
    """

    async def close(  # type:ignore[override]
        self, timeout: float = 30.0, reason: str = "nanny-close"
    ) -> Literal["OK"]:
        return await super().close(timeout=timeout, reason=reason)


def terminate_process(
    process: Type[BaseProcess], kill_wait: Union[float, int] = 3.0
) -> None:
    """
    Ensure a spawned process is terminated.

    Ensure a spawned process is really terminated to prevent the parent process
    (such as pytest) from freezing upon exit.

    Parameters
    ----------
    process:
        The process to be terminated.
    kill_wait: float or integer
        Maximum time to wait for the kill signal to terminate the process.

    Raises
    ------
    RuntimeError
        If the process terminated with a non-zero exit code.
    ValueError
        If the process was still alive after ``kill_wait`` seconds.
    """
    # Ensure process doesn't remain alive and hangs pytest
    if process.is_alive():
        process.kill()

    start_time = time.monotonic()
    while time.monotonic() - start_time < kill_wait:
        if not process.is_alive():
            break

    if process.is_alive():
        process.close()
    elif process.exitcode != 0:
        raise RuntimeError(
            f"Process did not exit cleanly (exit code: {process.exitcode})"
        )
