"""
KGRelationshipAgent mit AutoGen 4.7.2
Baut detaillierten Knowledge Graph via LLM auf
LLM-first mit Heuristic Fallback
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.autogen_config import get_autogen_config, get_agent_config, get_relationship_types, get_node_types

logger = logging.getLogger(__name__)


def _norm_key(s: str) -> str:
    """Normalize string for ID generation"""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9äöüß\s:\-_/\.]", "", s)
    return s


def _entity_id(prefix: str, name: str) -> str:
    """Generate entity ID from prefix and name"""
    return f"{prefix}:{_norm_key(name).replace(' ', '_')}"


class KGRelationshipAgent:
    """
    KGRelationshipAgent mit AutoGen 4.7.2
    Baut detaillierten Knowledge Graph via LLM auf
    """
    
    def __init__(self, heuristic_fallback: bool = True):
        """
        Initialisiert den KGRelationshipAgent

        Args:
            heuristic_fallback: Use heuristic extraction when LLM fails (default: True)
        """
        self.heuristic_fallback = heuristic_fallback
        self.config = get_agent_config("kg_relationship")
        
        # AutoGen 4.7.2 Model Client erstellen
        self.model_client = OpenAIChatCompletionClient(
            model=self.config["model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            seed=42,
            temperature=self.config["temperature"],
        )
        
        # AutoGen-Agent erstellen
        self.assistant = AssistantAgent(
            name="kg_relationship_assistant",
            system_message="""You are a knowledge graph expert. Your task is to extract relationships between requirements.
            
            Relationship types:
            - DEPENDS_ON: Requirement A depends on Requirement B
            - CONFLICTS_WITH: Requirement A conflicts with Requirement B
            - GROUPED_WITH: Requirements belong to a group
            - ACTOR_RELATION: Actor interacts with Requirement
            - DATA_FLOW: Data flow between Requirements
            - IMPLEMENTED_BY: Requirement implements Feature
            
            Node types:
            - Requirement: A requirement
            - Actor: A user or system actor
            - Entity: A database, API, or other entity
            - Tag: A category (e.g., Security, Functional, Performance)
            - Group: A feature group
            
            Return the knowledge graph in JSON format with nodes and edges.""",
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
    
    async def extract_relationships(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extrahiert Beziehungen zwischen Requirements
        LLM-first mit Heuristic Fallback

        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements

        Returns:
            Dict[str, Any]: Knowledge Graph mit Nodes und Edges
        """
        if not requirements:
            return {"nodes": [], "edges": []}

        # 1. LLM-first: Try LLM extraction
        try:
            kg = await self._llm_extract_relationships(requirements)

            # Check if LLM returned useful results
            if kg.get("nodes") or kg.get("edges"):
                logger.info(f"LLM extraction successful: {len(kg.get('nodes', []))} nodes, {len(kg.get('edges', []))} edges")
                return kg

            logger.warning("LLM returned empty graph, trying heuristic fallback")

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

        # 2. Heuristic fallback
        if self.heuristic_fallback:
            logger.info("Using heuristic fallback for KG extraction")
            return self._heuristic_extract(requirements)

        return {"nodes": [], "edges": []}

    async def _llm_extract_relationships(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM-based relationship extraction

        Args:
            requirements: List of requirements

        Returns:
            Knowledge graph with nodes and edges
        """
        # Requirements-Text für den Prompt vorbereiten
        requirements_text = "\n".join([
            f"{req.get('id', req.get('req_id', 'unknown'))}: {req.get('text', req.get('title', ''))}"
            for req in requirements
        ])

        prompt = f"""Extract relationships between the following requirements:

        {requirements_text}

        Available relationship types: {', '.join(get_relationship_types())}
        Available node types: {', '.join(get_node_types())}

        Return the knowledge graph in JSON format with nodes and edges."""

        cancellation_token = CancellationToken()
        response = await self.assistant.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token
        )

        # Extrahierte Beziehungen aus der Antwort parsen
        kg = self._parse_knowledge_graph(response.chat_message.content)

        return kg
    
    async def build_knowledge_graph(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Baut einen detaillierten Knowledge Graph auf
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            Dict[str, Any]: Knowledge Graph mit Nodes und Edges
        """
        # Beziehungen extrahieren
        kg = await self.extract_relationships(requirements)
        
        # Nodes und Edges validieren
        validated_kg = await self._validate_knowledge_graph(kg)
        
        return validated_kg
    
    async def _validate_knowledge_graph(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert den Knowledge Graph
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Validierter Knowledge Graph
        """
        # Nodes validieren
        validated_nodes = []
        for node in kg.get("nodes", []):
            is_valid = True
            issues = []
            
            # Prüfen, ob alle Pflichtfelder vorhanden sind
            required_fields = ["id", "type", "text"]
            for field in required_fields:
                if field not in node:
                    is_valid = False
                    issues.append(f"Missing required field: {field}")
            
            # Prüfen, ob der Node-Typ gültig ist
            valid_node_types = get_node_types()
            if node.get("type") not in valid_node_types:
                is_valid = False
                issues.append(f"Invalid node type: {node.get('type')}")
            
            # Validierungsergebnis hinzufügen
            node["is_valid"] = is_valid
            node["validation_issues"] = issues
            
            validated_nodes.append(node)
        
        # Edges validieren
        validated_edges = []
        for edge in kg.get("edges", []):
            is_valid = True
            issues = []
            
            # Prüfen, ob alle Pflichtfelder vorhanden sind
            required_fields = ["source", "target", "type"]
            for field in required_fields:
                if field not in edge:
                    is_valid = False
                    issues.append(f"Missing required field: {field}")
            
            # Prüfen, ob der Beziehungstyp gültig ist
            valid_relationship_types = get_relationship_types()
            if edge.get("type") not in valid_relationship_types:
                is_valid = False
                issues.append(f"Invalid relationship type: {edge.get('type')}")
            
            # Validierungsergebnis hinzufügen
            edge["is_valid"] = is_valid
            edge["validation_issues"] = issues
            
            validated_edges.append(edge)
        
        return {
            "nodes": validated_nodes,
            "edges": validated_edges
        }
    
    def _parse_knowledge_graph(self, response: str) -> Dict[str, Any]:
        """
        Parst den Knowledge Graph aus der LLM-Antwort
        
        Args:
            response (str): LLM-Antwort
        
        Returns:
            Dict[str, Any]: Knowledge Graph
        """
        try:
            # Versuchen, JSON aus der Antwort zu extrahieren
            if "extract_relationships" in response:
                # Function-Calling-Antwort parsen
                import re
                match = re.search(r'"nodes":\s*\[(.*?)\]', response, re.DOTALL)
                if match:
                    nodes_json = f"[{match.group(1)}]"
                    nodes = json.loads(nodes_json)
                else:
                    nodes = []
                
                match = re.search(r'"edges":\s*\[(.*?)\]', response, re.DOTALL)
                if match:
                    edges_json = f"[{match.group(1)}]"
                    edges = json.loads(edges_json)
                else:
                    edges = []
                
                return {"nodes": nodes, "edges": edges}
            
            # Fallback: Versuchen, JSON direkt zu parsen
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Leeren Knowledge Graph zurückgeben
            return {"nodes": [], "edges": []}

    def _heuristic_extract(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heuristic-based KG extraction (fallback when LLM fails)
        Extracts Actor/Entity/Action from requirement text using keyword matching

        Args:
            requirements: List of requirements with id/text fields

        Returns:
            Knowledge graph with nodes and edges
        """
        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []

        for req in requirements:
            req_id = str(req.get("id") or req.get("req_id") or "").strip()
            text = str(req.get("text") or req.get("title") or "").strip()
            tag = str(req.get("tag") or "functional").strip()

            if not req_id or not text:
                continue

            # Requirement Node
            req_node = {
                "id": req_id,
                "type": "Requirement",
                "text": text,
                "tag": tag,
            }
            all_nodes.append(req_node)

            # Tag Node + Edge
            tag_id = _entity_id("Tag", tag)
            all_nodes.append({
                "id": tag_id,
                "type": "Tag",
                "text": tag,
            })
            all_edges.append({
                "source": req_id,
                "target": tag_id,
                "type": "HAS_TAG",
            })

            # Heuristic: Actor detection
            lower = text.lower()
            actor = None
            if "benutzer" in lower:
                actor = "Benutzer"
            elif "nutzer" in lower:
                actor = "Nutzer"
            elif "user" in lower:
                actor = "User"
            elif "admin" in lower:
                actor = "Admin"
            elif "system" in lower:
                actor = "System"

            if actor:
                actor_id = _entity_id("Actor", actor)
                all_nodes.append({
                    "id": actor_id,
                    "type": "Actor",
                    "text": actor,
                })
                all_edges.append({
                    "source": req_id,
                    "target": actor_id,
                    "type": "ACTOR_RELATION",
                })

            # Heuristic: Entity detection
            entity_keywords = [
                "profil", "passwort", "token", "rollen", "rolle", "account",
                "formular", "suchergebnis", "deployment", "metriken", "datenbank",
                "api", "schnittstelle", "dokument", "datei", "email"
            ]
            for keyword in entity_keywords:
                if keyword in lower:
                    entity_id = _entity_id("Entity", keyword.capitalize())
                    all_nodes.append({
                        "id": entity_id,
                        "type": "Entity",
                        "text": keyword.capitalize(),
                    })
                    all_edges.append({
                        "source": req_id,
                        "target": entity_id,
                        "type": "DATA_FLOW",
                    })
                    break  # Only first entity per requirement

            # Heuristic: Action detection (German verbs ending in 'en')
            tokens = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", text)
            for t in tokens:
                tl = t.lower()
                if len(tl) > 3 and tl.endswith("en") and tl not in ["token", "rollen"]:
                    action_id = _entity_id("Action", t)
                    all_nodes.append({
                        "id": action_id,
                        "type": "Action",
                        "text": t,
                    })
                    all_edges.append({
                        "source": req_id,
                        "target": action_id,
                        "type": "HAS_ACTION",
                    })
                    break  # Only first action per requirement

        # Deduplicate nodes by ID
        seen_ids = set()
        unique_nodes = []
        for n in all_nodes:
            if n["id"] not in seen_ids:
                seen_ids.add(n["id"])
                unique_nodes.append(n)

        # Deduplicate edges
        seen_edges = set()
        unique_edges = []
        for e in all_edges:
            key = f"{e['source']}|{e['type']}|{e['target']}"
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        logger.info(f"Heuristic extraction: {len(unique_nodes)} nodes, {len(unique_edges)} edges")
        return {"nodes": unique_nodes, "edges": unique_edges}

    async def get_node_by_id(self, kg: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """
        Gibt einen Node anhand seiner ID zurück
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            node_id (str): Node ID
        
        Returns:
            Dict[str, Any]: Node
        """
        for node in kg.get("nodes", []):
            if node["id"] == node_id:
                return node
        
        return None
    
    async def get_edges_by_node(self, kg: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
        """
        Gibt alle Edges zurück, die mit einem Node verbunden sind
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            node_id (str): Node ID
        
        Returns:
            List[Dict[str, Any]]: Liste der Edges
        """
        edges = []
        for edge in kg.get("edges", []):
            if edge["source"] == node_id or edge["target"] == node_id:
                edges.append(edge)
        
        return edges
    
    async def get_dependencies(self, kg: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
        """
        Gibt alle Abhängigkeiten eines Nodes zurück
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            node_id (str): Node ID
        
        Returns:
            List[Dict[str, Any]]: Liste der Abhängigkeiten
        """
        dependencies = []
        for edge in kg.get("edges", []):
            if edge["source"] == node_id and edge["type"] == "DEPENDS_ON":
                dependencies.append(edge)
        
        return dependencies
    
    async def get_conflicts(self, kg: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
        """
        Gibt alle Konflikte eines Nodes zurück
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            node_id (str): Node ID
        
        Returns:
            List[Dict[str, Any]]: Liste der Konflikte
        """
        conflicts = []
        for edge in kg.get("edges", []):
            if edge["source"] == node_id and edge["type"] == "CONFLICTS_WITH":
                conflicts.append(edge)
        
        return conflicts
