"""Stress testing engine for local AI servers."""
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import httpx

from client import ModelClient, ChatMetrics
from logger import setup_logger, log_request_error, log_test_summary, log_vllm_error
from ui.components import estimate_tokens
from think_parser import split_thinking


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
    # New: real metrics from API
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft: float = 0.0
    decode_tps: float = 0.0

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
    # Decode-only metrics (excludes prefill)
    avg_decode_tps: float = 0.0
    avg_ttft: float = 0.0
    # Throughput (total tokens / wall clock)
    total_throughput_tps: float = 0.0
    wall_clock_time: float = 0.0
    total_output_tokens: int = 0
    # Percentiles
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    p50_ttft: float = 0.0
    p95_ttft: float = 0.0
    p50_decode_tps: float = 0.0
    # Sustained load
    max_concurrent: int = 0
    # True when any successful result fell back to estimate_tokens()
    # (server didn't return usage) — throughput numbers are approximate.
    throughput_estimated: bool = False
    # Consistency / variance metrics — useful whenever N >= 2 but primarily
    # consumed by the consistency test to isolate hardware-level noise.
    stddev_response_time: float = 0.0
    stddev_ttft: float = 0.0
    stddev_decode_tps: float = 0.0
    min_decode_tps: float = 0.0
    max_decode_tps: float = 0.0
    min_ttft: float = 0.0
    max_ttft: float = 0.0
    # First-half / second-half means (drift detection — thermal throttling,
    # driver warmup, etc.). 0.0 when fewer than 4 samples are available.
    first_half_decode_tps: float = 0.0
    second_half_decode_tps: float = 0.0
    first_half_ttft: float = 0.0
    second_half_ttft: float = 0.0
    # Free-text header for the run (set by consistency test prompt).
    notes: str = ""

    def add_error(self, error_msg: str):
        """Add error to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.errors.append(f"[{timestamp}] {error_msg}")

    @staticmethod
    def percentile(values: List[float], pct: float) -> float:
        """Calculate percentile from a sorted-ascending list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class StressTester:
    """Stress testing engine for AI models."""

    # Prompt banks — short for quick-fire, medium for realistic, long for context-heavy
    PROMPTS_SHORT = [
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
    ]

    PROMPTS_MEDIUM = [
        "Compare and contrast three different sorting algorithms — merge sort, quicksort, and heapsort. For each one, explain the core idea, the time complexity in best, average, and worst cases, the space complexity, and when you would choose it over the others in a real application.",
        "A small e-commerce company wants to migrate from a monolithic Django application to microservices. They have 50,000 daily active users, a PostgreSQL database, and a team of four developers. Outline a realistic migration strategy, including what to split first, how to handle data consistency, and the biggest risks they should watch for.",
        "Explain the difference between symmetric and asymmetric encryption. Then walk through how TLS 1.3 uses both during a handshake to establish a secure connection between a browser and a web server. Include what happens if the certificate is invalid.",
        "Write a detailed code review of the following Python function, identifying bugs, performance issues, and style problems:\n\ndef find_dupes(lst):\n    dupes = []\n    for i in range(len(lst)):\n        for j in range(len(lst)):\n            if i != j and lst[i] == lst[j]:\n                if lst[i] not in dupes:\n                    dupes.append(lst[i])\n    return dupes",
        "Describe how a CPU executes a single instruction from the moment it is fetched from memory to the moment the result is written back. Cover the fetch, decode, execute, memory access, and write-back stages. Explain what pipelining adds to this process and what hazards it introduces.",
        "A hospital needs a system to schedule operating rooms across three buildings. Surgeries have varying durations, some require specialized equipment, and emergency cases must preempt scheduled ones. Design the key data structures and the scheduling algorithm you would use, explaining the tradeoffs.",
        "Explain the CAP theorem using a concrete example. Then describe how Cassandra, MongoDB, and CockroachDB each make different tradeoffs within CAP, and what that means for an application developer choosing between them.",
        "Write a technical explanation of how garbage collection works in the JVM. Cover the generational hypothesis, the young and old generation spaces, the different collector algorithms (Serial, Parallel, G1, ZGC), and when each is appropriate.",
        "A data pipeline processes 10 million events per hour from Kafka, enriches them with data from a Redis cache, and writes the results to S3 in Parquet format. Latency has spiked from 200ms to 5 seconds per batch. Walk through a systematic debugging approach, listing what you would check first and why.",
        "Explain how attention mechanisms work in transformer models, starting from the basic dot-product attention, then multi-head attention, and finally how positional encoding lets the model understand token order. Use concrete examples with small matrices to illustrate the computation.",
    ]

    PROMPTS_LONG = [
        "You are given the following passage from a research paper abstract. Summarize it in three bullet points, then critique the methodology described:\n\n"
        "We present a novel approach to few-shot image classification that combines prototypical networks with a learned task-adaptive projection layer. Our method, TaskProj, trains a small MLP to project support set embeddings into a task-specific subspace before computing prototype distances. We evaluate on miniImageNet, tieredImageNet, and CUB-200 across 1-shot and 5-shot settings. TaskProj achieves state-of-the-art results on miniImageNet 1-shot (68.4% accuracy, +2.1% over ProtoNet) and competitive results on 5-shot. However, on CUB-200, performance degrades when support examples are visually similar, suggesting the projection layer may collapse fine-grained distinctions. We also observe that TaskProj requires 3x more training episodes to converge compared to the baseline. We hypothesize this is due to the additional parameters in the projection MLP competing with the backbone for gradient signal during meta-training. Ablation studies show that a single linear projection layer captures 89% of the improvement, raising questions about whether the MLP complexity is justified. Our code and pretrained models are available at the provided repository link.",
        "Analyze the following server access log entries and identify any security concerns, categorize the types of requests, and recommend specific mitigations:\n\n"
        "192.168.1.105 - - [10/Apr/2026:14:23:01 +0000] \"GET /api/users?id=1 OR 1=1-- HTTP/1.1\" 200 4523\n"
        "192.168.1.105 - - [10/Apr/2026:14:23:03 +0000] \"GET /api/users?id=1 UNION SELECT * FROM credentials-- HTTP/1.1\" 200 8921\n"
        "10.0.0.42 - - [10/Apr/2026:14:24:15 +0000] \"POST /api/login HTTP/1.1\" 401 89\n"
        "10.0.0.42 - - [10/Apr/2026:14:24:16 +0000] \"POST /api/login HTTP/1.1\" 401 89\n"
        "10.0.0.42 - - [10/Apr/2026:14:24:16 +0000] \"POST /api/login HTTP/1.1\" 401 89\n"
        "10.0.0.42 - - [10/Apr/2026:14:24:17 +0000] \"POST /api/login HTTP/1.1\" 401 89\n"
        "10.0.0.42 - - [10/Apr/2026:14:24:17 +0000] \"POST /api/login HTTP/1.1\" 200 1245\n"
        "203.0.113.77 - - [10/Apr/2026:14:25:00 +0000] \"GET /admin/../../etc/passwd HTTP/1.1\" 403 0\n"
        "203.0.113.77 - - [10/Apr/2026:14:25:02 +0000] \"GET /admin/%2e%2e%2f%2e%2e%2fetc/passwd HTTP/1.1\" 200 1847\n"
        "172.16.0.5 - - [10/Apr/2026:14:30:00 +0000] \"GET /health HTTP/1.1\" 200 15\n"
        "172.16.0.5 - - [10/Apr/2026:14:30:30 +0000] \"GET /metrics HTTP/1.1\" 200 8234\n"
        "Provide a detailed analysis of each group of requests, the attack vectors involved, why some succeeded, and specific countermeasures.",
        "Review the following database schema and query, identify performance problems, and rewrite the query with optimizations. Explain each change you make:\n\n"
        "Tables:\n"
        "  orders (id INT PK, user_id INT, status VARCHAR(20), created_at TIMESTAMP, total DECIMAL(10,2))\n"
        "  order_items (id INT PK, order_id INT FK->orders.id, product_id INT, quantity INT, price DECIMAL(10,2))\n"
        "  products (id INT PK, name VARCHAR(255), category_id INT, is_active BOOLEAN)\n"
        "  categories (id INT PK, name VARCHAR(100), parent_id INT SELF-FK)\n"
        "  users (id INT PK, email VARCHAR(255), created_at TIMESTAMP, country VARCHAR(2))\n\n"
        "Indexes: orders(user_id), products(category_id)\n\n"
        "Query:\n"
        "SELECT u.email, u.country, COUNT(DISTINCT o.id) as order_count,\n"
        "       SUM(oi.quantity * oi.price) as total_spent,\n"
        "       GROUP_CONCAT(DISTINCT c.name) as categories\n"
        "FROM users u\n"
        "JOIN orders o ON o.user_id = u.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "JOIN products p ON p.id = oi.product_id\n"
        "JOIN categories c ON c.id = p.category_id\n"
        "WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)\n"
        "  AND o.status != 'cancelled'\n"
        "  AND p.is_active = 1\n"
        "GROUP BY u.id\n"
        "HAVING total_spent > 100\n"
        "ORDER BY total_spent DESC\n"
        "LIMIT 1000;",
        "You are a technical architect reviewing a pull request that introduces a new caching layer. The PR description reads:\n\n"
        "\"Added Redis caching to the user profile endpoint. Cache TTL is 24 hours. Cache key is user:{user_id}. On profile update, we invalidate the cache by deleting the key. We chose Redis over Memcached because we might need sorted sets later for the leaderboard feature. The cache is a read-through pattern — on miss, we query PostgreSQL and populate the cache before returning.\"\n\n"
        "The endpoint serves 50,000 requests per minute at peak. Profile updates happen roughly 200 times per minute across all users. The PostgreSQL read replica has a replication lag of 100-500ms.\n\n"
        "Identify all potential issues with this caching strategy, including race conditions, consistency problems, failure modes, and scaling concerns. For each issue, explain the specific scenario that triggers it and propose a fix.",
        "Explain the following Rust code to someone who knows Python but has never seen Rust. Cover ownership, borrowing, lifetimes, and why the compiler rejects the commented-out lines:\n\n"
        "fn main() {\n"
        "    let mut data = vec![1, 2, 3, 4, 5];\n"
        "    let slice = &data[1..3];\n"
        "    println!(\"slice: {:?}\", slice);\n"
        "    // data.push(6);  // ERROR: cannot borrow `data` as mutable\n"
        "    drop(slice);\n"
        "    data.push(6);  // OK now\n\n"
        "    let first = &data[0];\n"
        "    let second = &data[1];\n"
        "    println!(\"{} {}\", first, second);\n"
        "    // let third = &mut data[2];  // ERROR: cannot borrow as mutable\n\n"
        "    process(&data);\n"
        "    data.push(7);\n"
        "}\n\n"
        "fn process(items: &[i32]) -> i32 {\n"
        "    items.iter().sum()\n"
        "}\n\n"
        "Explain what would happen in Python with equivalent operations on a list, and why Rust's approach prevents bugs that Python allows.",
    ]

    def __init__(
        self, server: dict, model: str,
        max_tokens: int = 256, system_prompt: str = ""
    ):
        """Initialize stress tester.

        Args:
            server: Server configuration dict
            model: Model name to test
            max_tokens: Max output tokens per request (controls fairness)
            system_prompt: Optional system prompt prepended to all requests
        """
        self.server = server
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.logger = setup_logger("stress_tester")
        self.results: List[TestResult] = []
        self.stats = TestStats()
        # Reused across every request in a run to preserve the httpx
        # connection pool (keepalive, no re-handshake per request).
        self._http_client: Optional[httpx.AsyncClient] = None
        # One ModelClient per run — it's cheap, and chat_with_metrics
        # doesn't mutate instance state (metrics returned via tuple).
        self._model_client = ModelClient(server, model)

    def _get_prompt(self, index: int, length: int = None) -> str:
        """Get test prompt by index, cycling through all banks.

        Args:
            index: Prompt index
            length: Optional target token length — selects from appropriate bank

        Returns:
            Test prompt string
        """
        if length is not None:
            # Select prompt from the bank closest to target length
            if length <= 200:
                bank = self.PROMPTS_SHORT
            elif length <= 2000:
                bank = self.PROMPTS_MEDIUM
            else:
                bank = self.PROMPTS_LONG
            prompt = bank[index % len(bank)]

            # If still shorter than target, pad with topically relevant expansion
            current_est = len(prompt) // 4  # rough token estimate
            if current_est < length:
                reps = (length - current_est) // 50
                prompt += ("\n\nAdditionally, explore related implications, historical context, "
                           "alternative viewpoints, and practical applications of the above. "
                           "Consider edge cases and potential counterarguments. ") * max(1, reps)
            return prompt

        # Default: cycle through all banks evenly
        all_prompts = self.PROMPTS_SHORT + self.PROMPTS_MEDIUM + self.PROMPTS_LONG
        return all_prompts[index % len(all_prompts)]

    async def _warmup(self, warmup_length: int = 0):
        """Send throwaway requests to warm up the model (KV cache, CUDA kernels, etc.).

        Args:
            warmup_length: If >= 2000, the second warmup request uses a
                medium-length prompt so KV-cache allocation for long
                sequences is amortized before the real run.

        Raises:
            RuntimeError: if both warmup attempts fail (server unreachable).
        """
        self.logger.info("Warming up model...")
        attempts = [("Hello", 1)]
        if warmup_length >= 2000:
            attempts.append((self.PROMPTS_MEDIUM[0], 32))
        else:
            attempts.append(("Hello", 1))

        failures: List[str] = []
        for prompt, mt in attempts:
            try:
                await self._model_client.chat_with_metrics(
                    prompt, [], max_tokens=mt, http_client=self._http_client
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                failures.append(msg)
                self.logger.warning(f"Warmup failed: {msg}")

        if len(failures) == len(attempts):
            raise RuntimeError(
                f"Warmup failed — server unreachable: {failures[0]}"
            )

    async def _run_single_request(
        self,
        request_id: int,
        prompt: str,
        update_callback: Callable = None
    ) -> TestResult:
        """Run a single test request with real API metrics.

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
            start_time=time.monotonic()
        )

        # Notify callback of start
        if update_callback:
            try:
                await update_callback(result)
            except Exception as e:
                self.logger.error(f"Callback error on start: {e}")

        try:
            self.logger.debug(f"Request #{request_id} starting: {prompt[:50]}...")

            # Build message history with optional system prompt
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            response, metrics = await self._model_client.chat_with_metrics(
                prompt, messages, max_tokens=self.max_tokens,
                http_client=self._http_client,
            )

            self.logger.debug(f"Request #{request_id} completed: {len(response)} chars")

            parsed = split_thinking(response)
            result.response = response
            result.end_time = time.monotonic()

            # Use real token counts from API, fall back to estimate
            result.completion_tokens = metrics.completion_tokens
            result.prompt_tokens = metrics.prompt_tokens
            result.ttft = metrics.ttft

            if metrics.completion_tokens > 0:
                result.token_count = metrics.completion_tokens
            else:
                result.token_count = estimate_tokens(parsed.content)

            # Legacy: total tok/s (wall time)
            if result.duration > 0 and result.token_count > 0:
                result.tokens_per_sec = result.token_count / result.duration

            # Decode tok/s: prefer Ollama's native timing, else derive from TTFT
            if metrics.eval_duration_ns > 0 and metrics.completion_tokens > 0:
                result.decode_tps = metrics.completion_tokens / (metrics.eval_duration_ns / 1e9)
            elif metrics.ttft > 0 and result.duration > metrics.ttft and result.token_count > 0:
                decode_time = result.duration - metrics.ttft
                result.decode_tps = result.token_count / decode_time
            else:
                result.decode_tps = result.tokens_per_sec  # best effort

            result.status = "success"
            self.stats.success += 1

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            result.end_time = time.monotonic()
            self.stats.failed += 1

            import traceback
            self.logger.error(f"Request #{request_id} failed: {type(e).__name__} - {str(e)}")
            self.logger.debug(traceback.format_exc())

            log_request_error(self.logger, request_id, e, {
                "server": f"{self.server['ip']}:{self.server['port']}",
                "model": self.model,
                "prompt_length": len(prompt)
            })

            error_msg = f"Request #{request_id}: {type(e).__name__} - {str(e)}"
            self.stats.add_error(error_msg)

        finally:
            self.stats.completed += 1

            if update_callback:
                try:
                    await update_callback(result)
                except Exception as e:
                    self.logger.error(f"Callback error on completion: {e}")

        return result

    def _compute_stats(self, results: List[TestResult], wall_clock: float = 0.0):
        """Compute aggregate statistics from results.

        Args:
            results: List of completed TestResult objects
            wall_clock: Total wall-clock time for the test (for throughput calc)
        """
        successful = [r for r in results if r.status == "success"]

        response_times = [r.duration for r in successful]
        tps_values = [r.tokens_per_sec for r in successful if r.tokens_per_sec > 0]
        decode_tps_values = [r.decode_tps for r in successful if r.decode_tps > 0]
        ttft_values = [r.ttft for r in successful if r.ttft > 0]

        # Averages
        if response_times:
            self.stats.avg_response_time = sum(response_times) / len(response_times)
        if tps_values:
            self.stats.avg_tps = sum(tps_values) / len(tps_values)
        if decode_tps_values:
            self.stats.avg_decode_tps = sum(decode_tps_values) / len(decode_tps_values)
        if ttft_values:
            self.stats.avg_ttft = sum(ttft_values) / len(ttft_values)

        # Throughput (total tokens across all requests / wall clock)
        total_tokens = sum(r.token_count for r in successful)
        self.stats.total_output_tokens = total_tokens
        self.stats.wall_clock_time = wall_clock
        if wall_clock > 0 and total_tokens > 0:
            self.stats.total_throughput_tps = total_tokens / wall_clock

        # Flag when any successful result fell back to estimate_tokens
        # (server returned no usage/eval_count). Throughput is approximate.
        self.stats.throughput_estimated = any(
            r.completion_tokens == 0 for r in successful
        )

        # Percentiles — suppress when N is too small for the estimate to be meaningful.
        # p50 at any N (single-value sample is still informative).
        # p95 needs N >= 10; p99 needs N >= 20.
        n = len(response_times)
        self.stats.p50_response_time = TestStats.percentile(response_times, 50)
        self.stats.p95_response_time = TestStats.percentile(response_times, 95) if n >= 10 else 0.0
        self.stats.p99_response_time = TestStats.percentile(response_times, 99) if n >= 20 else 0.0
        self.stats.p50_ttft = TestStats.percentile(ttft_values, 50)
        self.stats.p95_ttft = TestStats.percentile(ttft_values, 95) if len(ttft_values) >= 10 else 0.0
        self.stats.p50_decode_tps = TestStats.percentile(decode_tps_values, 50)

        # Variance / spread — needs at least 2 samples.
        import statistics
        if len(response_times) >= 2:
            self.stats.stddev_response_time = statistics.pstdev(response_times)
        if len(ttft_values) >= 2:
            self.stats.stddev_ttft = statistics.pstdev(ttft_values)
            self.stats.min_ttft = min(ttft_values)
            self.stats.max_ttft = max(ttft_values)
        if len(decode_tps_values) >= 2:
            self.stats.stddev_decode_tps = statistics.pstdev(decode_tps_values)
            self.stats.min_decode_tps = min(decode_tps_values)
            self.stats.max_decode_tps = max(decode_tps_values)

        # Drift detection — first-half vs second-half mean. Needs >=4 successes
        # so each half has >=2 samples. Results are ordered by request_id, which
        # for serial tests equals wall-clock order.
        if len(successful) >= 4:
            ordered = sorted(successful, key=lambda r: r.request_id)
            half = len(ordered) // 2
            first, second = ordered[:half], ordered[half:]
            first_dtps = [r.decode_tps for r in first if r.decode_tps > 0]
            second_dtps = [r.decode_tps for r in second if r.decode_tps > 0]
            first_ttft = [r.ttft for r in first if r.ttft > 0]
            second_ttft = [r.ttft for r in second if r.ttft > 0]
            if first_dtps and second_dtps:
                self.stats.first_half_decode_tps = sum(first_dtps) / len(first_dtps)
                self.stats.second_half_decode_tps = sum(second_dtps) / len(second_dtps)
            if first_ttft and second_ttft:
                self.stats.first_half_ttft = sum(first_ttft) / len(first_ttft)
                self.stats.second_half_ttft = sum(second_ttft) / len(second_ttft)

    def _finalize(self, mode: str) -> Optional[str]:
        """Log summary and persist results to disk.

        Returns the path to the persisted JSON file, or None on failure.
        """
        log_test_summary(self.logger, mode, asdict(self.stats))

        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"stress_results_{timestamp}.json"
            payload = {
                "mode": mode,
                "server": self.server.get("url"),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt,
                "timestamp": datetime.now().isoformat(),
                "stats": asdict(self.stats),
                "results": [asdict(r) for r in self.results],
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            self.logger.info(f"Results persisted to {path}")
            return str(path)
        except Exception as e:
            self.logger.error(f"Failed to persist results: {e}")
            return None

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
        self.logger.info(f"Target: {self.server['url']} | Model: {self.model} | max_tokens: {self.max_tokens}")

        self.results = []
        self.stats = TestStats(total=num_requests)

        # Shuffle prompt bank per run (seeded from monotonic_ns so each run is
        # different but we could log the seed if we ever needed reproducibility).
        # Append a unique per-request salt to defeat server-side prefix caching.
        all_prompts = self.PROMPTS_SHORT + self.PROMPTS_MEDIUM + self.PROMPTS_LONG
        rng = random.Random(time.monotonic_ns())
        shuffled = all_prompts[:]
        rng.shuffle(shuffled)

        async with httpx.AsyncClient(timeout=300.0) as http_client:
            self._http_client = http_client

            await self._warmup()

            tasks = []
            for i in range(num_requests):
                base = shuffled[i % len(shuffled)]
                prompt = f"{base}\n\n[req-{i + 1}]"
                task = self._run_single_request(i + 1, prompt, update_callback)
                tasks.append(task)

            wall_start = time.monotonic()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock = time.monotonic() - wall_start

        self._http_client = None

        for result in results:
            if isinstance(result, TestResult):
                self.results.append(result)

        self._compute_stats(self.results, wall_clock)

        self.logger.info(f"Throughput test complete: {self.stats.success}/{num_requests} successful, "
                         f"system throughput: {self.stats.total_throughput_tps:.1f} t/s")

        self._finalize("throughput")
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

        async with httpx.AsyncClient(timeout=300.0) as http_client:
            self._http_client = http_client

            await self._warmup(warmup_length=max(token_sizes) if token_sizes else 0)

            wall_start = time.monotonic()
            for i, size in enumerate(token_sizes):
                prompt = self._get_prompt(i, length=size)
                result = await self._run_single_request(i + 1, prompt, update_callback)
                self.results.append(result)
            wall_clock = time.monotonic() - wall_start

        self._http_client = None

        self._compute_stats(self.results, wall_clock)

        self.logger.info(f"Token stress test complete: {self.stats.success}/{len(token_sizes)} successful")

        self._finalize("token_stress")
        return self.stats

    async def run_consistency_test(
        self,
        prompt: str,
        iterations: int,
        notes: str = "",
        update_callback: Callable = None,
    ) -> TestStats:
        """Run the same prompt N times serially to isolate hardware-level variance.

        Everything model-side is held constant (identical prompt, max_tokens,
        system prompt). Remaining variance is attributable to the environment:
        DVFS, thermal throttling, driver hiccups, kernel scheduling noise, etc.

        Args:
            prompt: The prompt to send on every iteration (one fixed string).
            iterations: Number of serial runs.
            notes: Free-text header stored with the results (e.g. hardware
                conditions or test intent — "CPU boost off, fan 100%").
            update_callback: Optional progress callback.

        Returns:
            TestStats with the standard fields plus variance / drift stats.
        """
        self.logger.info(f"Starting consistency test: {iterations} serial runs")
        if notes:
            self.logger.info(f"Notes: {notes}")

        self.results = []
        self.stats = TestStats(total=iterations, notes=notes)

        # Pick warmup size roughly matched to the prompt (estimate_tokens is
        # cheap and imperfect but good enough to decide between short / medium).
        prompt_est_tokens = estimate_tokens(prompt)

        async with httpx.AsyncClient(timeout=300.0) as http_client:
            self._http_client = http_client

            await self._warmup(warmup_length=prompt_est_tokens)

            wall_start = time.monotonic()
            for i in range(iterations):
                result = await self._run_single_request(i + 1, prompt, update_callback)
                self.results.append(result)
            wall_clock = time.monotonic() - wall_start

        self._http_client = None

        self._compute_stats(self.results, wall_clock)
        # Preserve notes (TestStats was replaced in _compute_stats? no — same instance)
        self.stats.notes = notes

        self.logger.info(f"Consistency test complete: {self.stats.success}/{iterations} successful")

        self._finalize("consistency")
        return self.stats

    async def run_sustained_load_test(
        self,
        duration_minutes: int,
        requests_per_minute: int = 5,
        update_callback: Callable = None
    ) -> TestStats:
        """Run sustained load test with overlapping requests.

        Launches requests at the target rate regardless of whether previous
        requests have completed, allowing concurrent in-flight requests when
        the model cannot keep up.

        Args:
            duration_minutes: Test duration in minutes
            requests_per_minute: Number of requests to send per minute
            update_callback: Optional callback for progress updates

        Returns:
            TestStats object with results
        """
        interval = 60.0 / requests_per_minute
        total_requests = duration_minutes * requests_per_minute
        # Soft cap so a stalled server can't pile up unbounded open sockets.
        max_in_flight = max(20, requests_per_minute * 2)

        self.logger.info(f"Starting sustained load test: {duration_minutes}min @ {requests_per_minute} req/min")

        self.results = []
        self.stats = TestStats(total=total_requests)

        tasks = []
        in_flight = set()
        request_id = 0
        max_concurrent = 0
        stall_warned = False

        async with httpx.AsyncClient(timeout=300.0) as http_client:
            self._http_client = http_client

            await self._warmup()

            wall_start = time.monotonic()
            end_time = wall_start + (duration_minutes * 60)
            # Monotonic deadline: each launch targets wall_start + n*interval,
            # avoiding drift accumulation over long runs.
            next_fire = wall_start

            while time.monotonic() < end_time and request_id < total_requests:
                # Soft backpressure: if the server has stalled and tasks are
                # piling up, wait for one to finish before launching another.
                if len(in_flight) >= max_in_flight:
                    if not stall_warned:
                        self.logger.warning(
                            f"In-flight cap reached ({max_in_flight}) — server may be stalling. "
                            f"Pausing new launches until tasks drain."
                        )
                        stall_warned = True
                    await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)

                request_id += 1
                prompt = self._get_prompt(request_id - 1)

                task = asyncio.create_task(
                    self._run_single_request(request_id, prompt, update_callback)
                )
                tasks.append(task)
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)

                if len(in_flight) > max_concurrent:
                    max_concurrent = len(in_flight)

                if request_id < total_requests:
                    next_fire += interval
                    delay = next_fire - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

            # Wait for any in-flight requests to finish
            results = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock = time.monotonic() - wall_start

        self._http_client = None

        for result in results:
            if isinstance(result, TestResult):
                self.results.append(result)

        self._compute_stats(self.results, wall_clock)
        self.stats.max_concurrent = max_concurrent

        self.logger.info(f"Sustained load test complete: {self.stats.success}/{total_requests} successful, "
                         f"peak concurrency: {max_concurrent}")

        self._finalize("sustained_load")
        return self.stats
