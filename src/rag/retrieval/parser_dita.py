"""
DITA parsing pipeline for FAR and DFARS regulation source files.

Clones the official GSA regulation repositories and parses each DITA topic
file into a LlamaIndex Document with structured metadata for indexing.
"""

import os
import re
import json
from pathlib import Path
from git import Repo
from lxml import etree
from llama_index.core import Document

FAR_REPO_URL = "https://github.com/GSA/GSA-Acquisition-FAR.git"
DFARS_REPO_URL = "https://github.com/GSA/GSA-Acquisition-DFARS.git"


def clone_if_needed(url: str, path: Path) -> None:
    """
    Clone a git repository to path if it does not already exist locally.

    Args:
        url (str): Remote git URL to clone from.
        path (Path): Local destination path.
    """
    if not os.path.exists(path):
        Repo.clone_from(url, path)


def parse_ditamap(ditamap_path: Path, regulation: str) -> dict:
    """
    Parse a DITA map file and extract per-topic metadata.

    Args:
        ditamap_path (Path): Path to the .ditamap file.
        regulation (str): Regulation label, e.g. "FAR" or "DFARS".

    Returns:
        dict[str, dict]: Mapping of href -> {regulation, navtitle, fill_types}.
    """
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(str(ditamap_path), parser=parser)
    root = tree.getroot()

    topic_info = {}
    for topicref in root.xpath(".//topicref"):
        href = topicref.get("href")
        if not href:
            continue
        topic_info[href] = {
            "regulation": regulation,
            "navtitle": topicref.get("navtitle"),
            "fill_types": (topicref.get("xtrc") or "").split(),
        }
    return topic_info


def extract_fillins_and_text(dita_path: Path):
    """
    Extract full text, fill-in entries, and revision markers from a DITA file.

    Args:
        dita_path (Path): Path to the .dita topic file.

    Returns:
        tuple[str, list[dict], list[dict]]:
            - Full concatenated text of the topic.
            - List of fill-in dicts: {type, format, placeholder}.
            - List of revision marker dicts: {rev, text}.
    """
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(str(dita_path), parser=parser)
    root = tree.getroot()

    text = " ".join(root.itertext())

    fillins = []
    for cite in root.xpath(".//cite"):
        fillins.append({
            "type": cite.get("xtrf"),
            "format": cite.get("outputclass"),
            "placeholder": "".join(cite.itertext()).strip(),
        })

    rev_markers = []
    for el in root.xpath(".//*[@rev]"):
        rev_markers.append({
            "rev": el.get("rev"),
            "text": "".join(el.itertext()).strip(),
        })

    return text, fillins, rev_markers


def infer_part_subpart_section(href: str):
    """
    Infer FAR/DFARS structural identifiers from a DITA file href.

    Args:
        href (str): Relative href from the ditamap, e.g. "part-15.dita".

    Returns:
        tuple[str|None, str|None, str|None, str|None]: (part, subpart, section, clause)
    """
    name = Path(href).stem
    part = subpart = section = clause = None

    if name.startswith("part-"):
        part = name.split("part-")[1]
    elif name.startswith("subpart-"):
        subpart = name.split("subpart-")[1]
        part = subpart.split(".")[0]
    elif re.match(r"^\d+\.\d+(-\d+)?$", name):
        section = name
        part = name.split(".")[0]
    elif re.match(r"^\d+\.\d+-\d+", name):
        clause = name
        part = name.split(".")[0]

    return part, subpart, section, clause


def build_documents_from_repo(repo_path: str, regulation: str) -> list:
    """
    Parse all DITA topics in a regulation repository into LlamaIndex Documents.

    Args:
        repo_path (str): Local path to the cloned regulation repository.
        regulation (str): Regulation label, e.g. "FAR" or "DFARS".

    Returns:
        list[Document]: One Document per non-empty DITA topic, with metadata.
    """
    repo_path = Path(repo_path)
    ditamaps = list(repo_path.rglob("*.ditamap"))
    if not ditamaps:
        return []

    main_map = next(
        (dm for dm in ditamaps if dm.name.lower() == f"{regulation.lower()}.ditamap"),
        ditamaps[0],
    )
    topic_info = parse_ditamap(main_map, regulation)

    docs = []
    for href, info in topic_info.items():
        dita_file = (main_map.parent / href).resolve()
        if not dita_file.exists():
            continue

        text, fillins, rev_markers = extract_fillins_and_text(dita_file)
        if not text.strip():
            continue

        part, subpart, section, clause = infer_part_subpart_section(href)

        metadata = {
            "regulation": regulation,
            "href": href,
            "navtitle": info["navtitle"],
            "fill_types": info["fill_types"],
            "fillins": fillins,
            "rev_markers": rev_markers,
            "part": part,
            "subpart": subpart,
            "section": section,
            "clause": clause,
            "source_path": dita_file.relative_to(repo_path).as_posix() if dita_file.is_relative_to(repo_path) else href,
        }

        docs.append(Document(text=text, metadata=metadata))

    return docs
