# rag/retrieval/parser_dita.py
import os
import re
import json
from pathlib import Path
from git import Repo
from lxml import etree
from llama_index.core import Document

FAR_REPO_URL = "https://github.com/GSA/GSA-Acquisition-FAR.git"
DFARS_REPO_URL = "https://github.com/GSA/GSA-Acquisition-DFARS.git"

def clone_if_needed(url, path):
    if not os.path.exists(path):
        Repo.clone_from(url, path)

def parse_ditamap(ditamap_path: Path, regulation: str):
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

def build_documents_from_repo(repo_path: str, regulation: str):
    repo_path = Path(repo_path)
    ditamaps = list(repo_path.rglob("*.ditamap"))
    if not ditamaps:
        return []

    main_map = next((dm for dm in ditamaps if dm.name.lower() == f"{regulation.lower()}.ditamap"), ditamaps[0])
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
            "source_path": str(dita_file),
        }

        docs.append(Document(text=text, metadata=metadata))

    return docs
