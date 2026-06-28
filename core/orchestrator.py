"""
darkclaw/core/orchestrator.py
Main agent loop — the heart of Darkclaw.

The orchestrator is the one place that ties the whole system together:

  task in
    → pick an agent
    → inject Darkclaw memory as context (s05 Knowledge-on-Demand)
    → run the agent
    → if it fails, hand the failure to the HealEngine (self-healing)
    → learn from whatever came back (self-teaching)
    → every state change is published to the event_bus so both the
      operator dashboard AND Maple Creek light up in real time.

The loop (from the sanbuphy/learn-coding-agent analysis):
    while True:
        response = llm.completion(messages)
        if response.stop_reason == "tool_use":
            messages.append(dispatch_tool(response.tool_calls))
        else:
            break

In Darkclaw that inner loop lives inside each agent's .run(); the
orchestrator owns the *outer* loop: memory, healing, teaching, events.
"""
import asyncio
import os
import time

from core.event_bus import emit, EventType
from core.tool_registry import ToolRegistry
from agents.base_agent import TaskResult


DEFAULT_DB = os.environ.get(
    "DARKCLAW_DB",
    os.path.expanduser("~/darkclaw/data/darkclaw.db"),
)


class Orchestrator:
    """
    Owns the agent registry, the tool registry, and the outer task loop.

    Build a fully-wired system with::

        orc = Orchestrator.bootstrap()          # memory + heal + teach + agents + guardian
        result = await orc.submit("What is Book Burrow's address?")
    """

    def __init__(self, memory=None, heal_engine=None, teach_engine=None, tools=None):
        self.memory = memory
        self.healer = heal_engine
        self.teacher = teach_engine
        self.tools = tools or ToolRegistry()
        self.agents = {}
        self.guardian = None
        self.task_queue = []
        self._task_ctr = 0

    # ── wiring ──────────────────────────────────────────────────────────

    def register_agent(self, agent):
        self.agents[agent.agent_id] = agent
        if self.guardian:
            self.guardian.register(agent)
        emit(EventType.AGENT_STARTED, agent.agent_id,
             role=agent.config.role, model=agent.config.model)
        return agent

    def attach_guardian(self, guardian):
        self.guardian = guardian
        for agent in self.agents.values():
            guardian.register(agent)
        return guardian

    # ── the outer loop ──────────────────────────────────────────────────

    async def submit(self, task: str, agent_id: str = None, context: dict = None) -> TaskResult:
        """
        Submit a task. Returns a TaskResult.

        Flow: pick agent → inject memory → run → heal-on-failure → teach.
        """
        context = dict(context or {})
        target_id = agent_id or self._route(task)
        if not target_id:
            raise RuntimeError("No agents registered")
        agent = self.agents[target_id]

        self._task_ctr += 1
        context.setdefault("task_text", task)
        context.setdefault("fallback_model", agent.config.fallback_model)

        # ── inject Darkclaw memory (s05) ────────────────────────────────
        if self.memory and "injected_memory" not in context:
            qr = self.memory.query(task, target_id)
            context["injected_memory"] = qr.to_tool_result()
            emit(EventType.MEMORY_QUERY, target_id,
                 query=task[:80], tier=qr._classify_tier(), method=qr.method)

        retry_fn = lambda: agent.run(task, context)

        # ── run, then self-heal if it breaks ────────────────────────────
        try:
            result = await agent.run(task, context)
            if not result.success and result.error:
                raise RuntimeError(result.error)
        except Exception as e:
            result = await self._heal(target_id, e, context, retry_fn)

        # ── self-teach from a good result ───────────────────────────────
        if self.teacher and result and result.success and result.output:
            self.teacher.ingest_from_text(target_id, str(result.output))

        return result

    async def _heal(self, agent_id, error, context, retry_fn) -> TaskResult:
        """Hand a failure to the HealEngine and translate the outcome back to a TaskResult."""
        if not self.healer:
            return TaskResult(success=False, output=None, agent_id=agent_id,
                              duration_ms=0.0, error=str(error))

        repair = await self.healer.heal(agent_id, error, context, retry_fn)
        if self.teacher:
            self.teacher.ingest_heal_signal(agent_id, repair)

        if repair.success:
            if isinstance(repair.output, TaskResult):
                return repair.output
            return TaskResult(success=True, output=repair.output,
                              agent_id=agent_id, duration_ms=0.0)
        return TaskResult(success=False, output=None, agent_id=agent_id,
                          duration_ms=0.0, error=str(error))

    # ── default tools ───────────────────────────────────────────────────

    def _register_default_tools(self):
        """Register the built-in memory tools so agents can read/write Darkclaw."""
        mem = self.memory
        if not mem:
            return

        async def memory_query(inp: dict) -> dict:
            res = mem.query(inp.get("query", ""), inp.get("agent_id", "main"))
            return res.to_tool_result()

        async def memory_teach(inp: dict) -> dict:
            fact = mem.ingest_fact(
                inp.get("agent_id", "main"),
                inp["subject"], inp["predicate"], inp["object"],
            )
            triple = f"{inp['subject']} {inp['predicate']} {inp['object']}"
            if fact:
                emit(EventType.MEMORY_INGEST, inp.get("agent_id", "main"),
                     subject=inp["subject"], predicate=inp["predicate"], object=inp["object"])
            return {"ingested": bool(fact), "fact": triple}

        self.tools.register({
            "name": "memory_query",
            "description": "Search Darkclaw memory for a fact about an entity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language question"},
                    "agent_id": {"type": "string"},
                },
                "required": ["query"],
            },
            "handler": memory_query,
        })
        self.tools.register({
            "name": "memory_teach",
            "description": "Store a fact in Darkclaw memory as a subject-predicate-object triple.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "agent_id": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            },
            "handler": memory_teach,
        })

    # ── health summary (for UI /health) ─────────────────────────────────

    def health(self) -> dict:
        return {
            "agents": {aid: a.health() for aid, a in self.agents.items()},
            "memory": self.memory.stats() if self.memory else {},
            "heal": self.healer.stats() if self.healer else {},
            "teach": self.teacher.stats() if self.teacher else {},
            "tasks_submitted": self._task_ctr,
        }

    # ── bootstrap: build a fully-wired system ───────────────────────────

    @classmethod
    def bootstrap(cls, db_path: str = DEFAULT_DB, with_default_agents: bool = True):
        """
        Construct memory + heal + teach + tools + agents + guardian, all wired.
        This is the one call ui/server.py needs.
        """
        import os
        from memory.darkclaw_core import DarkclawEngine
        from core.heal_engine import HealEngine
        from core.teach_engine import TeachEngine

        # Wire OLLAMA_API_BASE so litellm can reach the configured Ollama server
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        if ollama_url:
            os.environ["OLLAMA_API_BASE"] = ollama_url

        model = cls._resolve_model()

        memory = DarkclawEngine(db_path=db_path)
        healer = HealEngine(memory_engine=memory)
        teacher = TeachEngine(memory_engine=memory)

        orc = cls(memory=memory, heal_engine=healer, teach_engine=teacher)
        healer.orc = orc
        orc._register_default_tools()

        if with_default_agents:
            from agents.worker import WorkerAgent
            from agents.coder  import CoderAgent
            from agents.base_agent import AgentConfig

            ollama = "ollama/"
            claude_coder = "claude-sonnet-4-6"

            # Rosie — fastest model, general helper / chat
            orc.register_agent(WorkerAgent(
                AgentConfig(agent_id="rosie", role="helper",
                            model=f"{ollama}phi3.5:latest"), memory=memory))

            # Kit — Book Burrow shopkeeper
            orc.register_agent(WorkerAgent(
                AgentConfig(agent_id="kit", role="shopkeeper",
                            model=f"{ollama}llama3.2:latest"), memory=memory))

            # Sage — system/homelab queries; uses trained weights
            orc.register_agent(WorkerAgent(
                AgentConfig(agent_id="sage", role="memory",
                            model=f"{ollama}openclaw-brain-v2:latest"), memory=memory))

            # Bea — general worker
            orc.register_agent(WorkerAgent(
                AgentConfig(agent_id="bea", role="worker",
                            model=f"{ollama}llama3.1:latest"), memory=memory))

            # Coder — local deepseek + Claude teacher until graduation
            orc.register_agent(CoderAgent(
                AgentConfig(agent_id="coder", role="coder",
                            model=f"{ollama}deepseek-coder-v2:16b-lite-instruct-q4_K_M",
                            teacher_model=claude_coder), memory=memory))

        from agents.guardian import GuardianAgent
        orc.attach_guardian(GuardianAgent(orchestrator=orc, heal_engine=healer))

        return orc

    def _route(self, task: str) -> str:
        """Auto-select the best agent for a task based on keyword signals."""
        if not self.agents:
            return None

        words = set(task.lower().split())

        _CODING = {"code", "function", "bug", "script", "python", "javascript",
                   "error", "fix", "implement", "debug", "refactor", "class",
                   "import", "syntax", "compile", "exception", "traceback", "def"}
        _SYSTEM = {"server", "docker", "proxmox", "gpu", "nvidia", "homelab",
                   "linux", "disk", "cpu", "ram", "memory", "ollama", "container",
                   "systemd", "service", "log", "process", "network", "tailscale",
                   "vm", "lxc", "bios", "kernel", "driver", "nvme", "pcie"}
        _SHOP   = {"book", "shop", "customer", "inventory", "order", "isbn",
                   "price", "supplier", "burrow", "sale", "invoice", "stock",
                   "vendor", "receipt", "return", "shelf", "catalog"}

        if words & _CODING and "coder" in self.agents:
            return "coder"
        if words & _SYSTEM and "sage" in self.agents:
            return "sage"
        if words & _SHOP and "kit" in self.agents:
            return "kit"

        # Default to Rosie (fastest, general helper)
        return "rosie" if "rosie" in self.agents else next(iter(self.agents))

    @staticmethod
    def _resolve_model() -> str:
        """Pick the best available model based on configured env vars."""
        import os
        dm = os.environ.get("DEFAULT_MODEL", "").strip()
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "")

        if dm:
            # Bare Ollama model name (e.g. "llama3.1:latest") — add litellm prefix
            if ollama_url and not dm.startswith(("claude", "gpt", "xai/", "gemini/", "mistral/")):
                return f"ollama/{dm}"
            return dm

        # Auto-select by priority: Anthropic → OpenAI → xAI → Gemini → Mistral → Ollama
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "claude-haiku-4-5"
        if os.environ.get("OPENAI_API_KEY"):
            return "gpt-4o-mini"
        if os.environ.get("XAI_API_KEY"):
            return "xai/grok-3-mini"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini/gemini-2.0-flash-lite"
        if os.environ.get("MISTRAL_API_KEY"):
            return "mistral/mistral-small-latest"
        if ollama_url:
            return "ollama/llama3.1:latest"
        return "claude-haiku-4-5"

    # ── demo scenario: exercise every subsystem for real ────────────────

    async def run_demo(self):
        """
        A scripted scenario that drives the REAL subsystems (memory, heal,
        teach) so both the dashboard and Maple Creek respond to live system
        state — not canned animations. Triggered by POST /demo.
        """
        emit(EventType.SYSTEM_START, "darkclaw", msg="Demo scenario starting")

        # 1. Real facts ingested into Darkclaw → Sage opens chests
        facts = [
            ("rosie", "Business_License", "HAS_STATUS", "Filed"),
            ("rosie", "Book_Burrow", "HAS_ADDRESS", "316_S_Buell_St"),
            ("sage",  "Book_Burrow", "LLC_NUMBER", "1580647"),
            ("sage",  "Book_Burrow", "HAS_STATUS", "Open"),
            ("bea",   "Book_Burrow", "SUPPLIED_BY", "Ingram"),
        ]
        for aid, s, p, o in facts:
            self.memory.ingest_fact(aid, s, p, o)
            emit(EventType.MEMORY_INGEST, aid, subject=s, predicate=p, object=o)
            await asyncio.sleep(0.8)

        # 2. A real task through a worker that hits rosie's own memory (GREEN tier)
        await self.submit("What is the status of the business license?", agent_id="rosie")
        await asyncio.sleep(0.6)

        # 3. A real failure → real HealEngine recovery (Pip alerts, Hero fights)
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("supplier API timed out")
            return TaskResult(success=True, agent_id="kit", duration_ms=12.0,
                              output="Reorder placed with Ingram")

        try:
            raise TimeoutError("supplier API timed out")
        except Exception as e:
            repair = await self.healer.heal("kit", e, {"task_text": "reorder stock"}, flaky)
            if self.teacher:
                self.teacher.ingest_heal_signal("kit", repair)
        await asyncio.sleep(0.6)

        # 4. A real eval that improves → TEACH_WIN (Sage celebrates)
        self.teacher.run_eval("sage", [("What is the capital of France?", "Paris")])
        await asyncio.sleep(0.3)
        self.teacher.run_eval("sage", [
            ("What is the LLC number for Book Burrow?", "1580647"),
            ("What is the status of Book Burrow?", "Open"),
        ])
        await asyncio.sleep(0.3)

        emit(EventType.AGENT_TASK_DONE, "rosie", task="demo complete")
        emit(EventType.SYSTEM_STOP, "darkclaw", msg="Demo scenario complete")
