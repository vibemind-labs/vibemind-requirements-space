"""
AutoGen-Konfiguration
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


def get_autogen_config() -> Dict[str, Any]:
    """
    Lädt die AutoGen-Konfiguration aus der JSON-Datei
    
    Returns:
        Dict[str, Any]: AutoGen-Konfiguration
    """
    config_path = Path("config/autogen_config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Environment-Variablen ersetzen
    config["api_key"] = os.getenv("OPENAI_API_KEY", config["api_key"])
    config["api_base"] = os.getenv("OPENAI_API_BASE", config["api_base"])
    
    return config


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Lädt die Konfiguration für einen spezifischen Agent
    
    Args:
        agent_name (str): Name des Agents (z.B. "requirements_extraction")
    
    Returns:
        Dict[str, Any]: Agent-Konfiguration
    """
    config = get_autogen_config()
    
    if agent_name not in config["agents"]:
        raise ValueError(f"Agent {agent_name} ist nicht in der Konfiguration definiert")
    
    return config["agents"][agent_name]


def get_chunking_config() -> Dict[str, Any]:
    """
    Lädt die Chunking-Konfiguration
    
    Returns:
        Dict[str, Any]: Chunking-Konfiguration
    """
    config = get_autogen_config()
    return config["chunking"]


def get_diagram_types() -> list:
    """
    Lädt die Diagramm-Typen
    
    Returns:
        list: Diagramm-Typen
    """
    config = get_autogen_config()
    return config["diagram_types"]


def get_relationship_types() -> list:
    """
    Lädt die Beziehungstypen
    
    Returns:
        list: Beziehungstypen
    """
    config = get_autogen_config()
    return config["relationship_types"]


def get_node_types() -> list:
    """
    Lädt die Node-Typen
    
    Returns:
        list: Node-Typen
    """
    config = get_autogen_config()
    return config["node_types"]


def get_validation_config() -> Dict[str, Any]:
    """
    Lädt die Validierungskonfiguration
    
    Returns:
        Dict[str, Any]: Validierungskonfiguration
    """
    config = get_autogen_config()
    return config["validation"]
