"""Network scanner for discovering local AI model servers."""
import asyncio
import socket
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import httpx

from config import config


async def check_endpoint(client: httpx.AsyncClient, url: str, endpoint: str) -> Optional[Dict]:
    """Check if an endpoint is available and return its type."""
    try:
        response = await client.get(f"{url}{endpoint}", timeout=2.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


async def probe_server(ip: str, port: int, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """Probe a single IP:port combination for AI model servers."""
    async with semaphore:  # Limit concurrent probes
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

        return None


async def scan_network(progress_callback=None, timeout: float = 30.0) -> List[Dict]:
    """Scan the local network for AI model servers.

    Args:
        progress_callback: Optional callback for progress updates
        timeout: Maximum time in seconds for entire scan (default: 30s)

    Returns:
        List of discovered servers
    """
    # Get local IP to determine subnet
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.1"

    # Extract subnet (e.g., 192.168.1.x)
    subnet = ".".join(local_ip.split(".")[:3])

    # Create shared resources for all probes
    semaphore = asyncio.Semaphore(50)  # Reduced from 100 to 50 for better stability

    # Shared HTTP client with timeouts and connection limits
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=25)
    timeout_config = httpx.Timeout(2.0, connect=1.0)  # 2s total, 1s connect

    try:
        async with httpx.AsyncClient(limits=limits, timeout=timeout_config) as client:
            # Generate all IP:port combinations to check
            tasks = []
            total = 255 * len(config.COMMON_PORTS)
            current = 0

            for i in range(1, 256):
                ip = f"{subnet}.{i}"
                for port in config.COMMON_PORTS:
                    tasks.append(probe_server(ip, port, client, semaphore))

            # Execute all probes concurrently with overall timeout
            servers = []

            try:
                # Wrap in wait_for to enforce overall timeout
                async with asyncio.timeout(timeout):  # Python 3.11+ syntax
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            current += 1

                            if progress_callback:
                                await progress_callback(current, total)

                            if result:
                                # Add basic health info
                                result["status"] = "discovered"
                                servers.append(result)
                        except Exception:
                            # Skip failed individual probes
                            current += 1
                            if progress_callback:
                                await progress_callback(current, total)
                            continue
            except asyncio.TimeoutError:
                # Scan timeout reached, return what we have
                pass

    except Exception as e:
        # Client creation or other fatal error
        import logging
        logging.error(f"Network scan failed: {e}")
        return []

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


def save_cache(servers: List[Dict]) -> None:
    """Save discovered servers to cache file."""
    try:
        with open(config.CACHE_FILE, 'w') as f:
            json.dump(servers, f, indent=2)
    except Exception:
        pass  # Silently fail if cache can't be saved


def load_cache() -> Optional[List[Dict]]:
    """Load servers from cache file."""
    try:
        if config.CACHE_FILE.exists():
            with open(config.CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass  # Silently fail if cache can't be loaded
    return None


def save_favorite(server: Dict, model: str) -> None:
    """Add a server/model combination to favorites."""
    favorites = load_favorites()

    # Create favorite entry
    favorite = {
        "server": server,
        "model": model,
        "added_at": int(asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0)
    }

    # Check if already in favorites
    for fav in favorites:
        if fav["server"]["url"] == server["url"] and fav["model"] == model:
            return  # Already favorited

    favorites.append(favorite)

    try:
        with open(config.FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=2)
    except Exception:
        pass


def load_favorites() -> List[Dict]:
    """Load favorites from file."""
    try:
        if config.FAVORITES_FILE.exists():
            with open(config.FAVORITES_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def remove_favorite(server_url: str, model: str) -> None:
    """Remove a favorite."""
    favorites = load_favorites()
    favorites = [f for f in favorites if not (f["server"]["url"] == server_url and f["model"] == model)]

    try:
        with open(config.FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=2)
    except Exception:
        pass


def is_favorite(server_url: str, model: str) -> bool:
    """Check if a server/model is favorited."""
    favorites = load_favorites()
    for fav in favorites:
        if fav["server"]["url"] == server_url and fav["model"] == model:
            return True
    return False


async def quick_validate_cache(servers: List[Dict], progress_callback=None) -> List[Dict]:
    """Quickly validate cached servers are still available."""
    validated = []
    total = len(servers)

    for i, server in enumerate(servers):
        if progress_callback:
            await progress_callback(i + 1, total)

        # Quick health check
        healthy_server = await check_server_health(server)
        if healthy_server.get("status") == "healthy":
            validated.append(healthy_server)

    return validated
