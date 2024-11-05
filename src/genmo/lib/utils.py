import os
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from genmo.lib.progress import get_new_progress_bar


class Timer:
    def __init__(self):
        self.times = {}  # Dictionary to store times per stage

    def __call__(self, name):
        print(f"Timing {name}")
        return self.TimerContextManager(self, name)

    def print_stats(self):
        total_time = sum(self.times.values())
        # Print table header
        print("{:<20} {:>10} {:>10}".format("Stage", "Time(s)", "Percent"))
        for name, t in self.times.items():
            percent = (t / total_time) * 100 if total_time > 0 else 0
            print("{:<20} {:>10.2f} {:>9.2f}%".format(name, t, percent))

    class TimerContextManager:
        def __init__(self, outer, name):
            self.outer = outer  # Reference to the Timer instance
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.outer.times[self.name] = self.outer.times.get(self.name, 0) + elapsed


def save_video(final_frames, output_path, fps=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(get_new_progress_bar(final_frames)):
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            frame_paths.append(frame_path)

        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        ffmpeg_cmd = (
            f"ffmpeg -y -r {fps} -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p -preset veryfast {output_path}"
        )
        try:
            subprocess.run(ffmpeg_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running ffmpeg:\n{e.stderr.decode()}")
import torch
from contextlib import ContextDecorator
from typing import Optional

class CudaTimer(ContextDecorator):
    """A context manager for timing CUDA operations with support for step-by-step timing and warmup steps.

    Args:
        name (str): Name of the timer for identification
        skip_steps (int, optional): Number of initial steps to skip (warmup). Defaults to 0.
        rank (int, optional): Process rank for distributed settings. Defaults to 0.
    """

    def __init__(self, name: str, skip_steps: int = 0, rank: int = 0):
        self.name = name
        self.skip_steps = skip_steps
        self.current_step = 0
        self.skipped_steps = 0
        self.recorded_steps = 0
        self.total_time = 0.0
        self.start_event = None
        self.end_event = None
        self.rank = rank

        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This timer requires CUDA support.")

    def __enter__(self):
        """Start timing by recording a CUDA event."""
        if self.rank == 0:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def step(self):
        """Record a timing step. Should be called after each operation to be timed."""
        if self.rank != 0:
            return

        self.current_step += 1

        # Handle warmup steps
        if self.current_step <= self.skip_steps:
            self.skipped_steps += 1
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            return

        # Record end event for timing
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        torch.cuda.synchronize()  # Ensure timing is accurate

        # Calculate and accumulate time
        if self.start_event is not None and self.end_event is not None:
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
            self.total_time += elapsed_time_ms
            self.recorded_steps += 1

        # Prepare for next step
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up and finalize timing."""
        if exc_type is not None:
            return False  # Re-raise any exceptions

        if self.rank == 0:
            torch.cuda.synchronize()

    @property
    def average_time(self) -> float:
        """Calculate average time per step in milliseconds."""
        if self.recorded_steps == 0:
            return 0.0
        return self.total_time / self.recorded_steps

    def summary(self) -> str:
        """Return a string summary of timing statistics."""
        return (f"Timer '{self.name}' summary:\n"
                f"Total time: {self.total_time:.2f}ms\n"
                f"Steps recorded: {self.recorded_steps}\n"
                f"Steps skipped: {self.skipped_steps}\n"
                f"Average time per step: {self.average_time:.2f}ms")