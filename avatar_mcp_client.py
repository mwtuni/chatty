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
        base = base_url or cfg.get("base_url") or os.getenv("AVATAR_MCP_BASE")
        self.base_url = (base or "http://localhost:7860").rstrip("/")
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
            resp = httpx.post(url, json=payload, timeout=10)
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
        lines.append("Always answer as this persona when speaking to the user.")
        return "\n".join(lines)
