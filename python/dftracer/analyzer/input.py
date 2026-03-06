import dataclasses as dc
from typing import Optional


@dc.dataclass
class FileInput:
    path: str


@dc.dataclass
class ZMQInput:
    address: str


@dc.dataclass
class MofkaInput:
    group_file: str
    topic_name: str
    control_topic_name: Optional[str] = None
