from dataclasses import dataclass, field

@dataclass
class Request:
    request_id: str
    category: str
    priority: str
    user: str
    status: str
    message: str = field(repr=False)

    def __post_init__(self) -> None:
        self._validate(
            self.request_id,
            self.category,
            self.priority,
            self.user,
            self.status,
            self.message
        )

    @staticmethod
    def _validate(
        request_id: str,
        category: str,
        priority: str,
        user: str,
        status: str,
        message: str
    ) -> None:
        if not request_id:
            raise ValueError("Пустой id")

        if not category:
            raise ValueError("Пустой category")

        if priority not in ("low", "medium", "high"):
            raise ValueError(f"Некорректный priority: {priority}")

        if not user:
            raise ValueError("Пустой user")

        if status not in ("open", "closed"):
            raise ValueError(f"Некорректный status: {status}")

        if not message:
            raise ValueError("Пустой message")