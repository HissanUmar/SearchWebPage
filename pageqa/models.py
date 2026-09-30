from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    path: str      # "Hooks > useState > Updating state"
    anchor: str    # "" if the section has no id
    text: str
    code: list = field(default_factory=list)


@dataclass
class Card:
    title: str
    h1: str
    intro: str
    outline: list
    purpose: str = ""


@dataclass
class Page:
    url: str
    title: str
    chunks: list
    card: Card = None
    embs: object = None


@dataclass
class Source:
    chunk: Chunk
    link: str


@dataclass
class Answer:
    purpose: str
    explanation: str
    sources: list
    found: bool
    top: list = field(default_factory=list)  # passages sent to the model (used by evals)
