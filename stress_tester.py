"""Stress testing engine for local AI servers."""
import asyncio
import time
from typing import Callable, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

from client import ModelClient
from logger import setup_logger, log_request_error, log_vllm_error
from utils.token_estimator import estimate_tokens


@dataclass
class TestResult:
    """Result of a single test request."""
    request_id: int
    status: str  # "pending", "running", "success", "error"
    prompt: str
    response: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    tokens_per_sec: float = 0.0
    error_msg: str = ""

    @property
    def duration(self) -> float:
        """Calculate request duration."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_complete(self) -> bool:
        """Check if request is complete."""
        return self.status in ("success", "error")


@dataclass
class TestStats:
    """Overall test statistics."""
    total: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    avg_response_time: float = 0.0
    avg_tps: float = 0.0
    errors: List[str] = field(default_factory=list)

    def add_error(self, error_msg: str):
        """Add error to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.errors.append(f"[{timestamp}] {error_msg}")


class StressTester:
    """Stress testing engine for AI models."""

    # Prompt bank for varied testing
    PROMPTS = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the main causes of climate change?",
        "Describe the plot of Romeo and Juliet.",
        "How do neural networks learn?",
        "What is the meaning of life?",
        "Explain the theory of relativity.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
        "How does photosynthesis work?",
        "What is blockchain technology?",
        "Explain supply and demand economics.",
        "What causes the Northern Lights?",
        "Describe the human digestive system.",
        "How do computers process information?",
        "What is the difference between AI and ML?",
        "Explain how vaccines work.",
        "What are the layers of Earth's atmosphere?",
        "Describe the process of DNA replication.",
    ]

    def __init__(self, server: dict, model: str):
        """Initialize stress tester.

        Args:
            server: Server configuration dict
            model: Model name to test
        """
        self.server = server
        self.model = model
        self.logger = setup_logger("stress_tester")
        self.results: List[TestResult] = []
        self.stats = TestStats()

    def _get_prompt(self, index: int, length: int = None) -> str:
        """Get test prompt by index.

        Args:
            index: Prompt index
            length: Optional token length for padding

        Returns:
            Test prompt string
        """
        prompt = self.PROMPTS[index % len(self.PROMPTS)]

        if length:
            # Pad prompt to approximate token count (rough estimate: 1 token ~= 4 chars)
            current_tokens = len(prompt) // 4
            if current_tokens < length:
                padding = "Please provide a detailed explanation. " * ((length - current_tokens) // 8)
                prompt = f"{prompt} {padding}"

        return prompt


    async def _run_single_request(
        self,
        request_id: int,
        prompt: str,
        update_callback: Callable = None
    ) -> TestResult:
        """Run a single test request.

        Args:
            request_id: Unique request identifier
            prompt: Prompt to send
            update_callback: Optional callback for progress updates

        Returns:
            TestResult object
        """
        result = TestResult(
            request_id=request_id,
            status="running",
            prompt=prompt,
            start_time=time.time()
        )

        # Notify callback of start
        if update_callback:
            try:
                await update_callback(result)
            except Exception as e:
                self.logger.error(f"Callback error on start: {e}")

        try:
            # Create client and make request
            client = ModelClient(self.server, self.model)

            # Log request start
            self.logger.debug(f"Request #{request_id} starting: {prompt[:50]}...")

            response = await client.chat(prompt, [])

            # Log response received
            self.logger.debug(f"Request #{request_id} completed: {len(response)} chars")

            result.response = response
            result.end_time = time.time()
            result.token_count = estimate_tokens(response)

            if result.duration > 0 and result.token_count > 0:
                result.tokens_per_sec = result.token_count / result.duration

            result.status = "success"
            self.stats.success += 1

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            result.end_time = time.time()
            self.stats.failed += 1

            # Log the error with full traceback
            import traceback
            self.logger.error(f"Request #{request_id} failed: {type(e).__name__} - {str(e)}")
            self.logger.debug(traceback.format_exc())

            # Log to error tracking
            log_request_error(self.logger, request_id, e, {
                "server": f"{self.server['ip']}:{self.server['port']}",
                "model": self.model,
                "prompt_length": len(prompt)
            })

            error_msg = f"Request #{request_id}: {type(e).__name__} - {str(e)}"
            self.stats.add_error(error_msg)

        finally:
            self.stats.completed += 1

            # Notify callback of completion
            if update_callback:
                try:
                    await update_callback(result)
                except Exception as e:
                    self.logger.error(f"Callback error on completion: {e}")

        return result

    async def run_throughput_test(
        self,
        num_requests: int,
        update_callback: Callable = None
    ) -> TestStats:
        """Run throughput test with concurrent requests.

        Args:
            num_requests: Number of concurrent requests to spawn
            update_callback: Optional callback for progress updates

        Returns:
            TestStats object with results
        """
        self.logger.info(f"Starting throughput test with {num_requests} concurrent requests")
        self.logger.info(f"Target: {self.server['url']} | Model: {self.model}")

        self.results = []
        self.stats = TestStats(total=num_requests)

        # Create all tasks
        tasks = []
        for i in range(num_requests):
            prompt = self._get_prompt(i)
            task = self._run_single_request(i + 1, prompt, update_callback)
            tasks.append(task)

        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        response_times = []
        tps_values = []

        for result in results:
            if isinstance(result, TestResult):
                self.results.append(result)
                if result.status == "success":
                    response_times.append(result.duration)
                    if result.tokens_per_sec > 0:
                        tps_values.append(result.tokens_per_sec)

        # Calculate averages
        if response_times:
            self.stats.avg_response_time = sum(response_times) / len(response_times)
        if tps_values:
            self.stats.avg_tps = sum(tps_values) / len(tps_values)

        self.logger.info(f"Throughput test complete: {self.stats.success}/{num_requests} successful")

        return self.stats

    async def run_token_stress_test(
        self,
        token_sizes: List[int] = None,
        update_callback: Callable = None
    ) -> TestStats:
        """Run token stress test with varying prompt lengths.

        Args:
            token_sizes: List of token counts to test (default: [500, 1000, 2000, 5000])
            update_callback: Optional callback for progress updates

        Returns:
            TestStats object with results
        """
        if token_sizes is None:
            token_sizes = [500, 1000, 2000, 5000]

        self.logger.info(f"Starting token stress test with sizes: {token_sizes}")

        self.results = []
        self.stats = TestStats(total=len(token_sizes))

        for i, size in enumerate(token_sizes):
            prompt = self._get_prompt(i, length=size)
            result = await self._run_single_request(i + 1, prompt, update_callback)
            self.results.append(result)

        # Calculate averages
        response_times = [r.duration for r in self.results if r.status == "success"]
        tps_values = [r.tokens_per_sec for r in self.results if r.status == "success" and r.tokens_per_sec > 0]

        if response_times:
            self.stats.avg_response_time = sum(response_times) / len(response_times)
        if tps_values:
            self.stats.avg_tps = sum(tps_values) / len(tps_values)

        self.logger.info(f"Token stress test complete: {self.stats.success}/{len(token_sizes)} successful")

        return self.stats

    async def run_sustained_load_test(
        self,
        duration_minutes: int,
        requests_per_minute: int = 5,
        update_callback: Callable = None
    ) -> TestStats:
        """Run sustained load test over time.

        Args:
            duration_minutes: Test duration in minutes
            requests_per_minute: Number of requests to send per minute
            update_callback: Optional callback for progress updates

        Returns:
            TestStats object with results
        """
        total_requests = duration_minutes * requests_per_minute
        interval = 60.0 / requests_per_minute  # Seconds between requests

        self.logger.info(f"Starting sustained load test: {duration_minutes}min @ {requests_per_minute} req/min")

        self.results = []
        self.stats = TestStats(total=total_requests)

        request_id = 1
        for _ in range(total_requests):
            prompt = self._get_prompt(request_id - 1)

            # Run request
            result = await self._run_single_request(request_id, prompt, update_callback)
            self.results.append(result)

            request_id += 1

            # Wait for interval (don't wait after last request)
            if request_id <= total_requests:
                await asyncio.sleep(interval)

        # Calculate averages
        response_times = [r.duration for r in self.results if r.status == "success"]
        tps_values = [r.tokens_per_sec for r in self.results if r.status == "success" and r.tokens_per_sec > 0]

        if response_times:
            self.stats.avg_response_time = sum(response_times) / len(response_times)
        if tps_values:
            self.stats.avg_tps = sum(tps_values) / len(tps_values)

        self.logger.info(f"Sustained load test complete: {self.stats.success}/{total_requests} successful")

        return self.stats
