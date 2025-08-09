# FILE: README.md
#1
# RoseApp
SeedApp-style skeleton with **Gemini (public scaffold)** + **ROSE_PURE (private core)** + **Rose50 (integrator)**.

## Quickstart
1) `python -m venv venv && source venv/bin/activate` (Win: `.\venv\Scripts\Activate.ps1`)
2) `pip install -r requirements.txt`
3) Run tests: `pytest -q`
4) Dev server (mock core by default): `uvicorn app.main:app --reload`
5) Switch to live core:
   - Linux/macOS: `export USE_MOCK_CORE=false`
   - Windows PS: `$env:USE_MOCK_CORE='false'`
   - (Optional) `export SEED_CORE_FACADE_PATH=app.seed_core.adapters:SeedFacade`
   - `uvicorn app.main:app --reload`

Endpoints (default prefix `/api/v1`):
- GET `/api/v1/ping` → `{"status":"pong"}`
- POST `/api/v1/seed` body `{"value": <float>}` → `{"result": value*2, "message": "..."}`

No secrets. Public routes never import `app.seed_core.*` directly.

---

# FILE: requirements.txt
#1
fastapi
pydantic
uvicorn
structlog
python-dotenv
pytest
httpx

# FILE: .gitignore
#1
venv/
__pycache__/
*.pyc
*.pyo
*.DS_Store
.coverage
.pytest_cache/
.env

# FILE: app/__init__.py
#1
# intentionally empty

# FILE: app/api/__init__.py
#1
# intentionally empty

# FILE: app/api/routes/__init__.py
#1
# intentionally empty

# FILE: app/schemas/__init__.py
#1
# intentionally empty

# FILE: app/seed_core/__init__.py
#1
# intentionally empty

# FILE: app/config/__init__.py
#1
# intentionally empty

# FILE: app/config/loader.py
#1 Public, keyless settings + dependency getter (env-driven).
from __future__ import annotations
from functools import lru_cache
from pydantic import BaseModel, Field
import os

#2
class PublicSettings(BaseModel):
    service_name: str = Field(default=os.getenv("PUBLIC_SERVICE_NAME", "SeedApp Public API"))
    service_version: str = Field(default=os.getenv("PUBLIC_SERVICE_VERSION", "0.2.0"))
    api_prefix: str = Field(default=os.getenv("API_PREFIX", "/api/v1"))
    seed_core_facade_path: str = Field(
        default=os.getenv("SEED_CORE_FACADE_PATH", "app.seed_core.adapters:SeedFacade"),
        description="module:Class path",
    )
    use_mock_core: bool = Field(default=os.getenv("USE_MOCK_CORE", "true").lower() != "false")
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "info"))

#3
@lru_cache(maxsize=1)
def get_settings() -> PublicSettings:
    return PublicSettings()

# FILE: app/schemas/seed.py
#1
from pydantic import BaseModel, Field

#2
class SeedInput(BaseModel):
    value: float = Field(..., description="A public scalar value for processing.")

#3
class SeedOutput(BaseModel):
    result: float = Field(..., description="The processed scalar result.")
    message: str = Field("Processing successful.", description="Status message.")

# FILE: app/adapters/public_io.py
#1 Public mock adapter kept for compatibility (not used once DI is wired).
from app.schemas.seed import SeedInput, SeedOutput

#2
def process_seed_value(input_data: SeedInput) -> SeedOutput:
    mocked_result = input_data.value * 2.0
    return SeedOutput(result=mocked_result, message="Processing successful with mock data.")

# FILE: app/api/routes/ping.py
#1
from fastapi import APIRouter

#2
router = APIRouter()

#3
@router.get("/ping", tags=["Health"])
async def ping() -> dict:
    return {"status": "pong"}

# FILE: app/api/routes/seed.py
#1 Public route using DI boundary (no private imports here).
from fastapi import APIRouter, Depends
import structlog
from app.schemas.seed import SeedInput, SeedOutput
from app.integrations.wiring import SeedProcessorPort, get_seed_processor

#2
router = APIRouter()
log = structlog.get_logger(__name__)

#3
@router.post("/seed", response_model=SeedOutput, tags=["Seed"])
async def process_seed(
    input_data: SeedInput,
    processor: SeedProcessorPort = Depends(get_seed_processor),
) -> SeedOutput:
    #4
    log.info("seed_request_received", value=input_data.value)
    out = processor.seed(value=input_data.value)
    result = float(out.get("result", 0.0))
    message = str(out.get("message", "Processing successful."))
    log.info("seed_request_processed", result=result)
    return SeedOutput(result=result, message=message)

# FILE: app/integrations/__init__.py
#1
# intentionally empty

# FILE: app/integrations/wiring.py
#1 Safe public→private wiring via a typed port.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Callable, Iterable
import importlib
import logging
import re
from contextlib import contextmanager
from fastapi import Depends
from app.config.loader import PublicSettings, get_settings

#2
@runtime_checkable
class SeedProcessorPort(Protocol):
    def seed(self, *, value: float, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def redact(self, text: str) -> str: ...
    def health(self) -> Dict[str, Any]: ...

#3
_SENSITIVE_PATTERNS: Iterable[re.Pattern] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"\b(glue logic|harmonic (?:mean|resonance)|context fingerprint)\b",
        r"\b(adaptive pass limit|redundancy pruning|perspective (?:sequencing|order))\b",
        r"\b\(Do Not Share\)",
    ]
)
_REDACTION_TOKEN = "«redacted: private method»"

#4
def _defensive_redact(text: str) -> str:
    redacted = text
    for pat in _SENSITIVE_PATTERNS:
        redacted = pat.sub(_REDACTION_TOKEN, redacted)
    return redacted

#5
@dataclass
class Checkpoint:
    tag: str
    payload: Optional[Dict[str, Any]] = None

#6
@contextmanager
def rollback_checkpoint(tag: str):
    try:
        yield Checkpoint(tag=tag)
    except Exception:
        logging.getLogger("wiring.rollback").warning("rollback_triggered", extra={"tag": tag})
        raise

#7
def _load_facade_ctor(path: str) -> Callable[[], Any]:
    mod_name, _, cls_name = path.partition(":")
    if not mod_name or not cls_name:
        raise ValueError("Invalid SEED_CORE_FACADE_PATH; expected 'pkg.mod:Class'")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls

#8
class CoreSeedProcessorAdapter(SeedProcessorPort):
    def __init__(self, ctor: Callable[[], Any]):
        self._core = ctor()

    def seed(self, *, value: float, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with rollback_checkpoint("seed"):
            result: Dict[str, Any] = self._core.seed(value=value, meta=meta or {})
            return {
                "result": float(result.get("result", value * 2.0)),
                "message": _defensive_redact(str(result.get("message", "processed"))),
            }

    def redact(self, text: str) -> str:
        core_redact = getattr(self._core, "redact", None)
        return core_redact(text) if callable(core_redact) else _defensive_redact(text)

    def health(self) -> Dict[str, Any]:
        base = {"status": "ok", "version": "core"}
        core_health = getattr(self._core, "health", None)
        if callable(core_health):
            try:
                payload = core_health()
                if isinstance(payload, dict):
                    for k in ("status", "version"):
                        if k in payload:
                            base[k] = payload[k]
            except Exception:
                base["status"] = "degraded"
        return base

#9
class MockSeedProcessorAdapter(SeedProcessorPort):
    def seed(self, *, value: float, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"result": float(value) * 2.0, "message": "mock processed"}

    def redact(self, text: str) -> str:
        return _defensive_redact(text)

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "version": "mock"}

#10
def build_seed_processor(settings: PublicSettings) -> SeedProcessorPort:
    if settings.use_mock_core:
        return MockSeedProcessorAdapter()
    try:
        ctor = _load_facade_ctor(settings.seed_core_facade_path)
        return CoreSeedProcessorAdapter(ctor=ctor)
    except Exception:
        logging.getLogger("wiring").warning(
            "core_init_failed_falling_back_to_mock",
            extra={"facade": settings.seed_core_facade_path},
        )
        return MockSeedProcessorAdapter()

#11
def get_seed_processor(settings: PublicSettings = Depends(get_settings)) -> SeedProcessorPort:
    return build_seed_processor(settings)

# FILE: app/main.py
#1 Unified FastAPI entry using DI wiring.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import ping, seed
from app.config.loader import get_settings

#2
def build_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name, version=settings.service_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(ping.router, prefix=settings.api_prefix)
    app.include_router(seed.router, prefix=settings.api_prefix)
    return app

#3
app = build_app()

# FILE: app/seed_core/seed_os.py
#1 Abstract/ciphered wrappers (no secrets).
from __future__ import annotations
import logging
from typing import Any, Protocol

#2
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#3
class CipherProtocol(Protocol):
    def encode(self, data: Any) -> str: ...
    def decode(self, token: str) -> Any: ...

#4
class SeedOS:
    def __init__(self, cipher: CipherProtocol):
        self._cipher = cipher

    def wrap_seed(self, payload: Any) -> str:
        logger.debug("Wrapping seed payload.")
        return self._cipher.encode(payload)

    def unwrap_seed(self, token: str) -> Any:
        logger.debug("Unwrapping seed token.")
        return self._cipher.decode(token)

# FILE: app/seed_core/engine.py
#1 Deterministic transform.
from __future__ import annotations
import logging
from typing import Any, Dict

#2
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#3
class SeedEngine:
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        value = payload.get("value", 0)
        if not isinstance(value, (int, float)):
            raise ValueError("Payload 'value' must be numeric.")
        result = value * 2
        logger.debug("Engine execution: %s -> %s", value, result)
        return {"value": value, "result": result}

# FILE: app/seed_core/decoder.py
#1 Safe base64+JSON encode/decode + checksum.
from __future__ import annotations
import base64
import json
import hashlib
import logging
from typing import Any, Tuple

#2
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#3
def safe_encode(obj: Any) -> Tuple[str, str]:
    encoded_bytes = json.dumps(obj, sort_keys=True).encode()
    token = base64.urlsafe_b64encode(encoded_bytes).decode()
    checksum = hashlib.sha256(token.encode()).hexdigest()
    logger.debug("Encoded with checksum %s", checksum)
    return token, checksum

#4
def safe_decode(token: str, checksum: str) -> Any:
    expected_checksum = hashlib.sha256(token.encode()).hexdigest()
    if expected_checksum != checksum:
        logger.error("Checksum mismatch on decode.")
        raise ValueError("Checksum mismatch.")
    decoded_bytes = base64.urlsafe_b64decode(token.encode())
    obj = json.loads(decoded_bytes.decode())
    logger.debug("Decoded object successfully.")
    return obj

# FILE: app/seed_core/adapters.py
#1 Public facade used by integrator.
from __future__ import annotations
import hashlib
import logging
from typing import Any, Dict, Tuple
from .seed_os import SeedOS
from .decoder import safe_encode, safe_decode
from .engine import SeedEngine

#2
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#3
class SeedFacade:
    """
    Methods:
      - prepare(value) -> (token, checksum)
      - execute(token, checksum) -> {"result": {...}, "message": str}
      - run(value) -> {"token","checksum","result","message"}
      - seed(value, meta=None) -> {"result": float, "message": str}  # used by wiring
      - redact(text) -> str
      - health() -> dict
    """
    def __init__(self):
        cipher_type = type(
            "Cipher",
            (),
            {
                "encode": lambda _self, d: safe_encode(d)[0],
                "decode": lambda _self, t: safe_decode(t, hashlib.sha256(t.encode()).hexdigest()),
            },
        )
        self._seed_os = SeedOS(cipher=cipher_type())  # type: ignore
        self._engine = SeedEngine()

    def prepare(self, value: Any) -> Tuple[str, str]:
        payload = {"value": value}
        token, checksum = safe_encode(payload)
        logger.debug("Prepared token and checksum.")
        return token, checksum

    def execute(self, token: str, checksum: str) -> Dict[str, Any]:
        payload = safe_decode(token, checksum)
        result = self._engine.execute(payload)
        logger.debug("Executed token successfully.")
        return {"result": result, "message": "Execution successful."}

    def run(self, value: Any) -> Dict[str, Any]:
        token, checksum = self.prepare(value)
        execution_result = self.execute(token, checksum)
        return {"token": token, "checksum": checksum, **execution_result}

    # === Integrator-used surface ===
    def seed(self, *, value: float, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        _ = meta
        out = self._engine.execute({"value": float(value)})
        return {"result": float(out["result"]), "message": "Processing successful."}

    def redact(self, text: str) -> str:
        return text

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "version": "core"}

# FILE: app/api/server.py
#1 Thin shim for legacy run targets; prefer `uvicorn app.main:app --reload`.
import uvicorn
from app.main import app

#2
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# FILE: tests/test_routes_smoke.py
#1 Smoke tests via unified entry.
import pytest
from fastapi.testclient import TestClient
from app.main import app

#2
@pytest.fixture(scope="module")
def client():
    return TestClient(app)

#3
def test_ping(client: TestClient):
    r = client.get("/api/v1/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "pong"}

#4
def test_seed_post_200(client: TestClient):
    r = client.post("/api/v1/seed", json={"value": 12.3})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == pytest.approx(24.6, rel=1e-9)
    assert "message" in body

# FILE: tests/test_integration_routes.py
#1 Integration tests: DI boundary uses mock by default.
import os
import pytest
from fastapi.testclient import TestClient
from app.main import build_app

#2
os.environ["USE_MOCK_CORE"] = "true"
os.environ["SEED_CORE_FACADE_PATH"] = "app.seed_core.adapters:SeedFacade"

#3
def test_ping_and_seed_contract():
    app = build_app()
    client = TestClient(app)

    rp = client.get("/api/v1/ping")
    assert rp.status_code == 200
    assert rp.json() == {"status": "pong"}

    rs = client.post("/api/v1/seed", json={"value": 2.5})
    assert rs.status_code == 200
    body = rs.json()
    assert body["result"] == pytest.approx(5.0, rel=1e-9)
    assert isinstance(body["message"], str) and body["message"]

# FILE: scripts/run_local.ps1
#1
if (-not (Test-Path -Path "venv")) { python -m venv venv }
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload

# FILE: scripts/run_local.sh
#1
#!/bin/bash
if [ ! -d "venv" ]; then python -m venv venv; fi
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# FILE: reports/gemini_summary.md
#1
### Gemini Scaffold Summary
Public API contracts mounted under `/api/v1/*`. No secrets; smoke tests included.

# FILE: reports/rose_private_summary.md
#1
## ROSE Private Core Summary
Deterministic, keyless core with encode/decode and engine x2 rule. No env/fs/net.

# FILE: reports/rose50_integration.md
#1
## Rose50 Integration Summary
Public routes call a DI-provided `SeedProcessorPort`. Mock by default; flip with `USE_MOCK_CORE=false`.
