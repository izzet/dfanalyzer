import dataclasses as dc


@dc.dataclass
class FileInput:
    path: str


@dc.dataclass
class ZMQInput:
    address: str
