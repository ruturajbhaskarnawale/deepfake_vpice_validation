from abc import ABC, abstractmethod
from typing import Any, Dict, List
from backend.app.models.pydantic_models import ThreatSignal

class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The logical name of this agent."""
        pass

    @abstractmethod
    async def process(self, case_id: str, context: Dict[str, Any]) -> List[ThreatSignal]:
        """Processes the given case context and returns a list of detected ThreatSignals."""
        pass
