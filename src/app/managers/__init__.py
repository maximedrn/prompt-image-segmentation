"""Long-lived orchestrators (singletons).

Managers own resource life cycles: model instances, caches, worker
pools. Everything below (models, infrastructure) is stateless per call;
everything above (services, api) borrows resources from a manager.
"""

from app.managers.model_manager import ModelManager

__all__: list[str] = ["ModelManager"]
