import os
from typing import Any, Dict, Optional

import httpx


class AvatarMCPClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        avatar_id: Optional[str] = None,
        admin_id: Optional[str] = None,
        logger=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        # Resolve backend preference (default to Hugging Face Space)
        backend_pref = (
            (cfg.get("backend") or os.getenv("AVATAR_BACKEND") or "huggingface")
            .strip()
            .lower()
        )
        hf_base_cfg = cfg.get("hf_base_url") or os.getenv("AVATAR_MCP_HF_BASE")
        local_base_cfg = cfg.get("local_base_url") or os.getenv("AVATAR_MCP_LOCAL_BASE")
        # Explicit overrides take precedence
        base = (
            base_url
            or cfg.get("base_url")
            or os.getenv("AVATAR_MCP_BASE")
            or os.getenv("AVATAR_MCP_BASE_URL")
        )
        default_hf = "https://mwtuni-avatar-mcp.hf.space"
        default_local = "http://localhost:7865"
        if base:
            resolved_base = base
        elif backend_pref in ("local", "dev", "localhost"):
            resolved_base = local_base_cfg or default_local
        else:
            resolved_base = hf_base_cfg or default_hf
        self.base_url = resolved_base.rstrip("/")
        self.avatar_id = avatar_id or cfg.get("avatar_id") or os.getenv("AVATAR_ID")
        self.admin_id = admin_id or cfg.get("admin_id") or os.getenv("AVATAR_ADMIN_ID")
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            try:
                self.logger(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def _request(self, path: str, payload: Dict[str, Any]):
        url = f"{self.base_url}{path}"
        try:
            resp = httpx.post(
                url, json=payload, timeout=20, follow_redirects=True
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._log(f"[avatar_mcp] request failed: {exc}")
            return None

    def fetch_context(self, private: bool = True) -> Optional[Dict[str, Any]]:
        if not self.avatar_id:
            return None
        payload = {"avatar_id": self.avatar_id}
        mode = "public"
        if private and self.admin_id:
            payload["admin_id"] = self.admin_id
            payload["mode"] = "admin"
            mode = "admin"
        data = self._request("/mcp/get_avatar_context", payload)
        if data:
            self._log(f"[avatar_mcp] loaded context ({mode})")
        return data

    def summarize(self, max_mem: int = 3) -> Optional[str]:
        if not self.avatar_id:
            return None
        payload = {"avatar_id": self.avatar_id, "max_mem": max_mem}
        data = self._request("/mcp/summarize_avatar", payload)
        if data and isinstance(data, dict):
            summary = data.get("summary")
            if summary:
                self._log("[avatar_mcp] summary ready")
                return summary
        return None

    def retrieve_snippets(self, query: str, limit: int = 3) -> Optional[list[str]]:
        if not self.avatar_id:
            return None
        payload = {"avatar_id": self.avatar_id, "query": query, "limit": limit}
        data = self._request("/mcp/retrieve_snippets", payload)
        if data and isinstance(data, dict):
            return data.get("snippets") or []
        return None

    def store_memory(self, entry: str, private: bool = True):
        if not self.avatar_id or not self.admin_id:
            return None
        payload = {
            "avatar_id": self.avatar_id,
            "admin_id": self.admin_id,
            "entry": entry,
            "private": private,
        }
        return self._request("/mcp/store_avatar_memory", payload)

    @staticmethod
    def build_prompt(context: Optional[Dict[str, Any]]) -> Optional[str]:
        if not context:
            return None
        persona = context.get("persona") or context.get("description")
        description = context.get("description")
        lines = []
        if persona:
            lines.append(f"You are {persona}. Stay in this persona at all times.")
        if description:
            lines.append(f"Description: {description}.")
        memories = context.get("memory") or []
        if memories:
            pub = [m["entry"] for m in memories if not m.get("private")]
            priv = [m["entry"] for m in memories if m.get("private")]
            if pub:
                lines.append("Public knowledge: " + " | ".join(pub[-5:]))
            if priv:
                lines.append(
                    "Private insights (never state directly): " + " | ".join(priv[-5:])
                )
        if not lines:
            return None
        lines.append(
            "Always answer as this persona when speaking to the user. "
            "Use persona/background details only when they directly help answer the user's request; "
            "do not list your profile or projects unless explicitly asked. "
            "Prioritize the user's question, be concise, and ignore irrelevant background."
        )
        return "\n".join(lines)
