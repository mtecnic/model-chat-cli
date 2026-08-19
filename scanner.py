"""Network scanner for discovering local AI model servers."""
import asyncio
import socket
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import httpx


# Common ports for local AI model servers
COMMON_PORTS = [
    11434,  # Ollama
    1234,   # LM Studio
    5000,   # Flask/Custom servers
    8000,   # FastAPI/Custom servers
    8001,   # llama.cpp OpenAI server
    8080,   # Alternative HTTP
]

TCP_TIMEOUT = 0.5      # Seconds for the fast TCP pre-check
HTTP_TIMEOUT = 2.0     # Seconds for endpoint checks on live servers
PRESCAN_CONCURRENCY = 500


def _detect_local_ip() -> str:
    """Get this host's local IPv4 address via a UDP connect to public DNS."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "192.168.1.1"


async def tcp_probe(ip: str, port: int, semaphore: asyncio.Semaphore,
                    timeout: float = TCP_TIMEOUT) -> bool:
    """Quickly check if a TCP port is open."""
    async with semaphore:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), timeout=timeout
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            return False


async def check_endpoint(client: httpx.AsyncClient, url: str, endpoint: str) -> Optional[Dict]:
    """Check if an endpoint is available and return its type."""
    try:
        response = await client.get(f"{url}{endpoint}", timeout=HTTP_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


async def probe_server(ip: str, port: int, client: httpx.AsyncClient) -> Optional[Dict]:
    """Probe a single IP:port combination for AI model servers."""
    base_url = f"http://{ip}:{port}"

    # Check both endpoints in parallel for speed
    openai_data, ollama_data = await asyncio.gather(
        check_endpoint(client, base_url, "/v1/models"),
        check_endpoint(client, base_url, "/api/tags"),
        return_exceptions=True
    )

    # Handle any exceptions from gather
    if isinstance(openai_data, Exception):
        openai_data = None
    if isinstance(ollama_data, Exception):
        ollama_data = None

    # Prefer OpenAI-compatible endpoint if both are available
    if openai_data:
        return {
            "ip": ip,
            "port": port,
            "url": base_url,
            "type": "openai",
            "models": [m.get("id", "unknown") for m in openai_data.get("data", [])],
        }

    if ollama_data:
        return {
            "ip": ip,
            "port": port,
            "url": base_url,
            "type": "ollama",
            "models": [m.get("name", "unknown") for m in ollama_data.get("models", [])],
        }

    # Fallback: detect Ollama servers with no models pulled via /api/version
    if openai_data is None and ollama_data is None:
        version_data = await check_endpoint(client, base_url, "/api/version")
        if version_data:
            return {
                "ip": ip,
                "port": port,
                "url": base_url,
                "type": "ollama",
                "models": [],
                "ollama_version": version_data.get("version", "unknown"),
                "note": "Ollama running, no models pulled",
            }

    return None


async def scan_network(
    progress_callback=None,
    extra_ips: Optional[List[str]] = None,
    concurrency: int = PRESCAN_CONCURRENCY,
    tcp_timeout: float = TCP_TIMEOUT,
    ports: Optional[List[int]] = None,
) -> List[Dict]:
    """Scan the local network for AI model servers.

    Phase 1 ("tcp"): fast TCP pre-scan of all IP:port combinations.
    Phase 2 ("probe"): HTTP endpoint checks, only for live ports.
    """
    # Get local IP to determine subnet
    local_ip = _detect_local_ip()

    # Extract subnet (e.g., 192.168.1.x)
    subnet = ".".join(local_ip.split(".")[:3])

    # Build the full IP list: whole subnet + always-include localhost + any extras
    ips = [f"{subnet}.{i}" for i in range(1, 256)]
    if "127.0.0.1" not in ips:
        ips.append("127.0.0.1")
    for ip in (extra_ips or []):
        if ip not in ips:
            ips.append(ip)

    ports = ports or COMMON_PORTS
    combos = [(ip, port) for ip in ips for port in ports]
    total = len(combos)

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    # Phase 1: TCP pre-scan — dead ports fail in milliseconds,
    # so most of the network is eliminated without any HTTP traffic
    async def prescan_combo(ip: str, port: int):
        is_open = await tcp_probe(ip, port, semaphore, timeout=tcp_timeout)
        return ip, port, is_open

    live_combos = []
    current = 0
    for coro in asyncio.as_completed(prescan_combo(ip, port) for ip, port in combos):
        ip, port, is_open = await coro
        current += 1

        if progress_callback:
            await progress_callback(current, total, "tcp")

        if is_open:
            live_combos.append((ip, port))

    # Phase 2: HTTP probes, only against ports that passed the pre-scan
    servers: List[Dict] = []
    if live_combos:
        total = total + len(live_combos)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        async with httpx.AsyncClient(limits=limits) as client:
            for coro in asyncio.as_completed(probe_server(ip, port, client) for ip, port in live_combos):
                result = await coro
                current += 1

                if progress_callback:
                    await progress_callback(current, total, "probe")

                if result:
                    # Add basic health info (will be validated properly later if needed)
                    result["status"] = "discovered"
                    servers.append(result)

    return servers


async def check_server_health(server: Dict) -> Dict:
    """Check the health/responsiveness of a server."""
    url = server["url"]

    try:
        async with httpx.AsyncClient() as client:
            start = asyncio.get_event_loop().time()

            if server["type"] == "openai":
                response = await client.get(f"{url}/v1/models", timeout=5.0)
            else:  # ollama
                response = await client.get(f"{url}/api/tags", timeout=5.0)

            elapsed = asyncio.get_event_loop().time() - start

            return {
                **server,
                "status": "healthy" if response.status_code == 200 else "error",
                "response_time": round(elapsed * 1000, 2),  # ms
            }
    except Exception as e:
        return {
            **server,
            "status": "error",
            "error": str(e),
        }


# Cache file location
CACHE_FILE = Path.home() / ".model_chat_cache.json"


def save_cache(servers: List[Dict]) -> None:
    """Save discovered servers to cache file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(servers, f, indent=2)
    except Exception:
        pass  # Silently fail if cache can't be saved


def load_cache() -> Optional[List[Dict]]:
    """Load servers from cache file."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass  # Silently fail if cache can't be loaded
    return None


async def quick_validate_cache(servers: List[Dict], progress_callback=None) -> List[Dict]:
    """Quickly validate cached servers are still available (concurrently)."""
    total = len(servers)

    async def validate_with_index(i: int, server: Dict):
        healthy_server = await check_server_health(server)
        if progress_callback:
            await progress_callback(i, total, "validate")
        return i, healthy_server

    results = await asyncio.gather(
        *(validate_with_index(i + 1, server) for i, server in enumerate(servers))
    )

    validated = []
    for _, healthy_server in results:
        if healthy_server.get("status") == "healthy":
            validated.append(healthy_server)

    return validated
