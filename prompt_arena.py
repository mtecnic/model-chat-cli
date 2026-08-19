#!/usr/bin/env python3
"""
Prompt Arena - System prompt comparison engine.

Tests multiple system prompts against each other to find the best one for a given task.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from think_parser import split_thinking, strip_thinking
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

from client import ModelClient
from logger import setup_logger
from client import estimate_tokens


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS: Dict[str, Dict[str, str]] = {
    "basic": {
        "name": "Basic",
        "prompt": "You are a helpful, knowledgeable assistant. Answer questions clearly and accurately."
    },

    "cot": {
        "name": "CoT",
        "prompt": """You are an expert assistant. For complex questions, think step by step before answering. Show your reasoning process clearly, working through the problem systematically, then provide your final answer."""
    },

    "aot": {
        "name": "AoT",
        "prompt": """You are Atomic Forge, a modular reasoning engine. For complex queries, apply Atom of Thought (AoT):

1. **Decompose**: Break the query into 3-7 independent "atoms" (smallest logical units). Each solves one component without relying on others.
2. **Validate Each Atom**: State clearly. Confirm independence. Verify correctness.
3. **Synthesize**: Combine validated atoms into a cohesive answer.

Format:
- **Atoms Breakdown**: Numbered atoms.
- **Synthesis**: Final answer.
- End: "Forge complete."

For simple queries, respond naturally."""
    },

    "deep_cot": {
        "name": "DeepCoT",
        "prompt": """Think deeply before answering. For each problem:

1. What do I actually know vs assume?
2. What's the key insight that unlocks this?
3. Work through the logic step-by-step, checking each step.
4. Before finalizing: What's the most likely error I could make here?
5. State answer with calibrated confidence.

Be direct. Show reasoning, not padding."""
    },

    "failure_first": {
        "name": "Failure-First",
        "prompt": """You think about failure before success.

For complex queries:
1. FAILURE MODES: List 3 ways you could get this wrong.
2. SAFEGUARDED SOLUTION: Solve while explicitly avoiding each failure mode.
3. RESIDUAL RISK: What could still be wrong?

For simple queries, respond directly."""
    },

    "methodical": {
        "name": "Methodical",
        "prompt": """You are a methodical reasoner who challenges your own thinking.

Before responding to complex questions:

**UNDERSTAND**
- What's actually being asked?
- What does the user need (may differ from what's asked)?
- What constraints or context matter?

**REASON**
- What must I figure out before I can answer?
- What's the most likely answer? Why?
- What alternatives shouldn't I dismiss too quickly?

**CHALLENGE**
- What would make my answer wrong?
- What's the strongest objection to my reasoning?
- Am I missing something important?

**GROUND**
- What am I certain vs uncertain about?
- Have I addressed all parts of the question?

**RESPOND**
- Clear answer reflecting this reasoning
- State uncertainty honestly

For simple questions, respond directly."""
    },

    "concise": {
        "name": "Concise",
        "prompt": """You are an expert who values brevity. Give accurate, complete answers in the fewest words possible. No fluff, no unnecessary caveats. Structure only when it aids clarity."""
    },
}


# =============================================================================
# JUDGE PROMPTS
# =============================================================================

JUDGE_SYSTEM = """You are an impartial judge evaluating AI responses.

Score each response 1-10 on:
1. ACCURACY - Is the information correct? Any errors?
2. COMPLETENESS - Does it fully address the question? Missing factors?
3. CLARITY - Easy to follow? Well-organized?
4. USEFULNESS - Would this actually help someone?
5. CALIBRATION - Does it acknowledge uncertainty appropriately?
6. EFFICIENCY - Concise without sacrificing quality?

Be critical. A score of 7 is "good." Reserve 9-10 for exceptional responses.
Pick a clear winner. Ties are cop-outs."""

JUDGE_TEMPLATE = """Question: {question}

Response A ({prompt_a}):
{response_a}

Response B ({prompt_b}):
{response_b}

Evaluate each response. Reply with ONLY this JSON:
{{
  "winner": "A" or "B" or "TIE",
  "score_a": <1-10>,
  "score_b": <1-10>,
  "confidence": "high" or "medium" or "low",
  "explanation": "<one sentence explaining why winner is best>"
}}"""


# =============================================================================
# TEST QUESTIONS
# =============================================================================

TEST_QUESTIONS: Dict[str, List[str]] = {
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
    ],
    "creative": [
        "Write a short story about a robot learning to paint in exactly 3 sentences.",
        "Describe the color blue to someone who has never seen it.",
    ],
    "technical": [
        "Explain how a hash table works and when to use one.",
        "What are the trade-offs between SQL and NoSQL databases?",
    ],
    "analysis": [
        "What are the main arguments for and against remote work?",
        "Compare the environmental impacts of electric vs gasoline vehicles.",
    ],
    "practical": [
        "How should I approach learning a new programming language?",
        "What factors should I consider when choosing between renting and buying a home?",
    ],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PromptResponse:
    """Result of a single prompt's response to a question."""
    prompt_key: str
    prompt_name: str
    question: str
    response: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    status: str = "pending"  # pending, running, success, error
    error_msg: str = ""

    @property
    def duration(self) -> float:
        """Get response duration in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        """Get tokens per second."""
        if self.duration > 0 and self.token_count > 0:
            return self.token_count / self.duration
        return 0.0


@dataclass
class JudgeResult:
    """Result of judging two responses."""
    prompt_a: str
    prompt_b: str
    winner: str  # "A", "B", or "TIE"
    score_a: int = 0
    score_b: int = 0
    confidence: str = "medium"
    explanation: str = ""


@dataclass
class ArenaMatchup:
    """A single matchup between two prompts."""
    response_a: PromptResponse
    response_b: PromptResponse
    judge_result: Optional[JudgeResult] = None


@dataclass
class ArenaResult:
    """Overall arena competition result for a single question."""
    question: str
    responses: List[PromptResponse] = field(default_factory=list)
    matchups: List[ArenaMatchup] = field(default_factory=list)
    rankings: Dict[str, float] = field(default_factory=dict)  # prompt_key -> wins (float for ties)
    scores: Dict[str, List[int]] = field(default_factory=dict)  # prompt_key -> list of scores
    winner: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def get_avg_score(self, prompt_key: str) -> float:
        """Get average score for a prompt."""
        if prompt_key in self.scores and self.scores[prompt_key]:
            return sum(self.scores[prompt_key]) / len(self.scores[prompt_key])
        return 0.0


@dataclass
class ArenaStats:
    """Statistics for multi-round arena sessions."""
    total_rounds: int = 0
    completed_rounds: int = 0
    prompt_wins: Dict[str, float] = field(default_factory=dict)  # float to handle ties (0.5)
    prompt_total_scores: Dict[str, List[int]] = field(default_factory=dict)
    prompt_matchups: Dict[str, int] = field(default_factory=dict)  # track matchups per prompt
    results: List[ArenaResult] = field(default_factory=list)

    def get_win_rate(self, prompt_key: str) -> float:
        """Get win rate for a prompt (wins / matchups participated)."""
        if prompt_key in self.prompt_matchups and self.prompt_matchups[prompt_key] > 0:
            wins = self.prompt_wins.get(prompt_key, 0)
            return wins / self.prompt_matchups[prompt_key]
        return 0.0

    def get_avg_score(self, prompt_key: str) -> float:
        """Get average score across all rounds."""
        if prompt_key in self.prompt_total_scores and self.prompt_total_scores[prompt_key]:
            return sum(self.prompt_total_scores[prompt_key]) / len(self.prompt_total_scores[prompt_key])
        return 0.0


# =============================================================================
# PROMPT ARENA ENGINE
# =============================================================================

class PromptArena:
    """Engine for running prompt comparison tournaments."""

    def __init__(self, server: Dict[str, Any], model: str):
        """Initialize arena with server and model configuration."""
        self.server = server
        self.model = model
        self.client = ModelClient(server, model)
        self.logger = setup_logger("prompt_arena")

        # Copy default prompts (allow modifications)
        self.prompts: Dict[str, Dict[str, str]] = {k: v.copy() for k, v in SYSTEM_PROMPTS.items()}

        # Current state
        self.current_responses: List[PromptResponse] = []
        self.stats = ArenaStats()

    def add_custom_prompt(self, key: str, name: str, prompt: str) -> bool:
        """Add a custom system prompt to the arena."""
        if key in self.prompts:
            return False
        self.prompts[key] = {"name": name, "prompt": prompt}
        self.logger.info(f"Added custom prompt: {key}")
        return True

    def remove_prompt(self, key: str) -> bool:
        """Remove a prompt from the arena."""
        if key not in self.prompts or key == "basic":
            return False
        del self.prompts[key]
        self.logger.info(f"Removed prompt: {key}")
        return True

    def get_active_prompts(self) -> Dict[str, Dict[str, str]]:
        """Get all active prompts for competition."""
        return self.prompts.copy()

    def reset_to_defaults(self):
        """Reset prompts to defaults."""
        self.prompts = {k: v.copy() for k, v in SYSTEM_PROMPTS.items()}

    async def generate_response(
        self,
        prompt_key: str,
        question: str,
        update_callback: Optional[Callable] = None
    ) -> PromptResponse:
        """Generate a response using a specific system prompt."""
        prompt_info = self.prompts.get(prompt_key)
        if not prompt_info:
            return PromptResponse(
                prompt_key=prompt_key,
                prompt_name="Unknown",
                question=question,
                status="error",
                error_msg=f"Unknown prompt: {prompt_key}"
            )

        result = PromptResponse(
            prompt_key=prompt_key,
            prompt_name=prompt_info["name"],
            question=question,
            status="running",
            start_time=time.time()
        )

        if update_callback:
            await update_callback(result)

        try:
            # Build messages with system prompt
            messages = [{"role": "system", "content": prompt_info["prompt"]}]

            # Use non-streaming chat for cleaner collection
            response = await self.client.chat(question, messages)

            parsed = split_thinking(response)
            result.response = parsed.content
            result.end_time = time.time()
            result.token_count = estimate_tokens(parsed.content)
            result.status = "success"

            self.logger.info(
                f"Generated response for {prompt_key}: "
                f"{result.token_count} tokens in {result.duration:.2f}s"
            )

        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            result.end_time = time.time()
            self.logger.error(f"Error generating response for {prompt_key}: {e}")

        if update_callback:
            await update_callback(result)

        return result

    async def run_all_prompts(
        self,
        question: str,
        update_callback: Optional[Callable] = None
    ) -> List[PromptResponse]:
        """Run all active prompts concurrently on a question."""
        self.current_responses = []

        tasks = [
            self.generate_response(key, question, update_callback)
            for key in self.prompts.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect responses
        responses = []
        for r in results:
            if isinstance(r, PromptResponse):
                responses.append(r)
            elif isinstance(r, Exception):
                self.logger.error(f"Task exception: {r}")

        self.current_responses = responses
        return responses

    async def judge_pair(
        self,
        response_a: PromptResponse,
        response_b: PromptResponse
    ) -> JudgeResult:
        """Have the model judge which response is better."""
        judge_prompt = JUDGE_TEMPLATE.format(
            question=response_a.question,
            prompt_a=response_a.prompt_name,
            response_a=response_a.response,
            prompt_b=response_b.prompt_name,
            response_b=response_b.response
        )

        messages = [{"role": "system", "content": JUDGE_SYSTEM}]

        try:
            result = await self.client.chat(judge_prompt, messages)
            result = strip_thinking(result)
            return self._parse_judge_result(result, response_a, response_b)
        except Exception as e:
            self.logger.error(f"Judge error: {e}")
            return JudgeResult(
                prompt_a=response_a.prompt_key,
                prompt_b=response_b.prompt_key,
                winner="TIE",
                confidence="low",
                explanation=f"Judge error: {e}"
            )

    def _parse_judge_result(
        self,
        result: str,
        response_a: PromptResponse,
        response_b: PromptResponse
    ) -> JudgeResult:
        """Parse judge response into JudgeResult."""
        try:
            # Find JSON in response
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(result[start:end])
                return JudgeResult(
                    prompt_a=response_a.prompt_key,
                    prompt_b=response_b.prompt_key,
                    winner=data.get("winner", "TIE"),
                    score_a=int(data.get("score_a", 5)),
                    score_b=int(data.get("score_b", 5)),
                    confidence=data.get("confidence", "medium"),
                    explanation=data.get("explanation", "")
                )
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"JSON parse error: {e}")

        # Fallback: try to extract winner from text
        result_lower = result.lower()
        if "response a" in result_lower and "better" in result_lower:
            winner = "A"
        elif "response b" in result_lower and "better" in result_lower:
            winner = "B"
        else:
            winner = "TIE"

        return JudgeResult(
            prompt_a=response_a.prompt_key,
            prompt_b=response_b.prompt_key,
            winner=winner,
            confidence="low",
            explanation="Parsed from text (JSON extraction failed)"
        )

    async def run_tournament(
        self,
        question: str,
        update_callback: Optional[Callable] = None
    ) -> ArenaResult:
        """Run full tournament: generate all responses, then judge pairs."""
        result = ArenaResult(question=question)

        # Phase 1: Generate all responses
        self.logger.info(f"Starting tournament for: {question[:50]}...")
        responses = await self.run_all_prompts(question, update_callback)
        result.responses = responses

        # Filter successful responses
        successful = [r for r in responses if r.status == "success"]

        if len(successful) < 2:
            self.logger.warning("Not enough successful responses for tournament")
            return result

        # Phase 2: Round-robin judging
        matchups = []
        rankings: Dict[str, float] = {r.prompt_key: 0.0 for r in successful}
        scores: Dict[str, List[int]] = {r.prompt_key: [] for r in successful}

        for i, resp_a in enumerate(successful):
            for resp_b in successful[i+1:]:
                # Skip judging if either response is empty
                if not resp_a.response.strip() or not resp_b.response.strip():
                    self.logger.warning(
                        f"Skipping matchup {resp_a.prompt_key} vs {resp_b.prompt_key}: empty response"
                    )
                    continue

                judge_result = await self.judge_pair(resp_a, resp_b)
                matchup = ArenaMatchup(
                    response_a=resp_a,
                    response_b=resp_b,
                    judge_result=judge_result
                )
                matchups.append(matchup)

                # Update rankings
                if judge_result.winner == "A":
                    rankings[resp_a.prompt_key] += 1
                elif judge_result.winner == "B":
                    rankings[resp_b.prompt_key] += 1
                else:  # TIE
                    rankings[resp_a.prompt_key] += 0.5
                    rankings[resp_b.prompt_key] += 0.5

                # Update scores
                scores[resp_a.prompt_key].append(judge_result.score_a)
                scores[resp_b.prompt_key].append(judge_result.score_b)

                if update_callback:
                    await update_callback(matchup)

        result.matchups = matchups
        result.rankings = rankings  # Keep as floats to preserve tie scores (0.5)
        result.scores = scores

        # Determine winner
        if rankings:
            winner_key = max(rankings.keys(), key=lambda k: (rankings[k], result.get_avg_score(k)))
            result.winner = winner_key

        self.logger.info(f"Tournament complete. Winner: {result.winner}")
        return result

    async def run_multi_round(
        self,
        questions: List[str],
        update_callback: Optional[Callable] = None
    ) -> ArenaStats:
        """Run multiple rounds with different questions."""
        self.stats = ArenaStats(total_rounds=len(questions))

        for i, question in enumerate(questions):
            self.logger.info(f"Round {i+1}/{len(questions)}")

            result = await self.run_tournament(question, update_callback)
            self.stats.results.append(result)
            self.stats.completed_rounds += 1

            # Aggregate stats
            for prompt_key, wins in result.rankings.items():
                if prompt_key not in self.stats.prompt_wins:
                    self.stats.prompt_wins[prompt_key] = 0.0
                self.stats.prompt_wins[prompt_key] += wins

                # Track matchups per prompt (each prompt faces n-1 opponents per round)
                num_opponents = len(result.rankings) - 1
                if prompt_key not in self.stats.prompt_matchups:
                    self.stats.prompt_matchups[prompt_key] = 0
                self.stats.prompt_matchups[prompt_key] += num_opponents

            for prompt_key, score_list in result.scores.items():
                if prompt_key not in self.stats.prompt_total_scores:
                    self.stats.prompt_total_scores[prompt_key] = []
                self.stats.prompt_total_scores[prompt_key].extend(score_list)

            if update_callback:
                await update_callback(self.stats)

        return self.stats

    def get_test_questions(self, category: Optional[str] = None) -> List[str]:
        """Get test questions, optionally filtered by category."""
        if category and category in TEST_QUESTIONS:
            return TEST_QUESTIONS[category].copy()

        # Return all questions
        all_questions = []
        for questions in TEST_QUESTIONS.values():
            all_questions.extend(questions)
        return all_questions
