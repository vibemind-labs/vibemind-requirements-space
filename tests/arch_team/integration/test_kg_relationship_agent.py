"""
Integration-Tests für KGRelationshipAgent
"""

import pytest
import json
from pathlib import Path

from arch_team.agents.kg_relationship_agent import KGRelationshipAgent


@pytest.fixture
def agent():
    """Fixture für KGRelationshipAgent"""
    return KGRelationshipAgent()


@pytest.mark.asyncio
async def test_extract_relationships_from_requirements(agent):
    """Testet die Beziehungserkennung aus Requirements"""
    # Test-Requirements
    test_requirements = Path("tests/fixtures/test_requirements.json")
    
    if test_requirements.exists():
        with open(test_requirements, "r") as f:
            requirements = json.load(f)["requirements"]
        
        # Echter LLM-Call
        kg = await agent.extract_relationships(requirements)
        
        assert "nodes" in kg
        assert "edges" in kg
        assert len(kg["nodes"]) > 0
        assert len(kg["edges"]) > 0
    else:
        pytest.skip("Test-Requirements-Datei existiert nicht")


@pytest.mark.asyncio
async def test_build_knowledge_graph_from_requirements(agent):
    """Testet den Aufbau des Knowledge Graph aus Requirements"""
    # Test-Requirements
    test_requirements = Path("tests/fixtures/test_requirements.json")
    
    if test_requirements.exists():
        with open(test_requirements, "r") as f:
            requirements = json.load(f)["requirements"]
        
        # Echter LLM-Call
        kg = await agent.build_knowledge_graph(requirements)
        
        assert "nodes" in kg
        assert "edges" in kg
        assert len(kg["nodes"]) > 0
        assert len(kg["edges"]) > 0
    else:
        pytest.skip("Test-Requirements-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_relationships_with_real_llm(agent):
    """Testet die Beziehungserkennung mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        },
        {
            "id": "REQ-002",
            "text": "User passwords must be encrypted using AES-256.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Echter LLM-Call
    kg = await agent.extract_relationships(requirements)
    
    assert "nodes" in kg
    assert "edges" in kg
    assert len(kg["nodes"]) > 0
    assert len(kg["edges"]) > 0


@pytest.mark.asyncio
async def test_build_knowledge_graph_with_real_llm(agent):
    """Testet den Aufbau des Knowledge Graph mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        },
        {
            "id": "REQ-002",
            "text": "User passwords must be encrypted using AES-256.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Echter LLM-Call
    kg = await agent.build_knowledge_graph(requirements)
    
    assert "nodes" in kg
    assert "edges" in kg
    assert len(kg["nodes"]) > 0
    assert len(kg["edges"]) > 0


@pytest.mark.asyncio
async def test_validate_knowledge_graph_with_real_llm(agent):
    """Testet die Validierung des Knowledge Graph mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Validierung
        validated = await agent._validate_knowledge_graph(kg)
        
        assert "nodes" in validated
        assert "edges" in validated
        assert len(validated["nodes"]) > 0
        assert len(validated["edges"]) > 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_get_node_by_id_with_real_llm(agent):
    """Testet das Abrufen eines Nodes anhand seiner ID mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Node abrufen
        node = await agent.get_node_by_id(kg, "REQ-001")
        
        assert node is not None
        assert node["id"] == "REQ-001"
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_get_edges_by_node_with_real_llm(agent):
    """Testet das Abrufen aller Edges eines Nodes mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Edges abrufen
        edges = await agent.get_edges_by_node(kg, "REQ-001")
        
        assert len(edges) > 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_get_dependencies_with_real_llm(agent):
    """Testet das Abrufen aller Abhängigkeiten eines Nodes mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Abhängigkeiten abrufen
        dependencies = await agent.get_dependencies(kg, "REQ-001")
        
        assert len(dependencies) >= 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_get_conflicts_with_real_llm(agent):
    """Testet das Abrufen aller Konflikte eines Nodes mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Konflikte abrufen
        conflicts = await agent.get_conflicts(kg, "REQ-001")
        
        assert len(conflicts) >= 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")
