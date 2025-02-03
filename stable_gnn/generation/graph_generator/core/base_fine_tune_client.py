from abc import ABC, abstractmethod
from typing import Any


class BaseFineTuneClient(ABC):
    @abstractmethod
    def prepare_training_data(self, raw_data: str) -> Any:
        pass

    @abstractmethod
    def upload_training_data(self, training_data: Any) -> None:
        pass

    @abstractmethod
    def start_fine_tuning(self, job_name: str) -> None:
        pass

    @abstractmethod
    def check_fine_tuning_status(self) -> str:
        pass

    @abstractmethod
    def download_fine_tuned_model(self, save_path: str) -> bool:
        pass
