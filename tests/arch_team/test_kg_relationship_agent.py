"""
Unit-Tests für KGRelationshipAgent
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from arch_team.agents.kg_relationship_agent import KGRelationshipAgent


@pytest.fixture
def agent():
    """Fixture für KGRelationshipAgent"""
    return KGRelationshipAgent()


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Testet die Agent-Initialisierung"""
    assert agent is not None
    assert agent.assistant is not None
    assert agent.user_proxy is not None
    assert agent.extraction_tool is not None


@pytest.mark.asyncio
async def test_extract_relationships(agent):
    """Testet die Beziehungserkennung"""
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
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"},
            {"id": "Security", "type": "Tag", "text": "Security"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "Security", "type": "HAS_TAG"},
            {"source": "REQ-002", "target": "Security", "type": "HAS_TAG"}
        ]
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        kg = await agent.extract_relationships(requirements)
        
        assert "nodes" in kg
        assert "edges" in kg
        assert len(kg["nodes"]) > 0
        assert len(kg["edges"]) > 0


@pytest.mark.asyncio
async def test_build_knowledge_graph(agent):
    """Testet den Aufbau des Knowledge Graph"""
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
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"},
            {"id": "Security", "type": "Tag", "text": "Security"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "Security", "type": "HAS_TAG"},
            {"source": "REQ-002", "target": "Security", "type": "HAS_TAG"}
        ]
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        kg = await agent.build_knowledge_graph(requirements)
        
        assert "nodes" in kg
        assert "edges" in kg
        assert len(kg["nodes"]) > 0
        assert len(kg["edges"]) > 0


@pytest.mark.asyncio
async def test_validate_knowledge_graph(agent):
    """Testet die Validierung des Knowledge Graph"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "InvalidType", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "Security", "type": "HAS_TAG"},
            {"source": "REQ-002", "target": "Security", "type": "InvalidType"}
        ]
    }
    
    validated = await agent._validate_knowledge_graph(kg)
    
    assert "nodes" in validated
    assert "edges" in validated
    assert validated["nodes"][0]["is_valid"] == True
    assert validated["nodes"][1]["is_valid"] == False
    assert validated["edges"][0]["is_valid"] == True
    assert validated["edges"][1]["is_valid"] == False


@pytest.mark.asyncio
async def test_get_node_by_id(agent):
    """Testet das Abrufen eines Nodes anhand seiner ID"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": []
    }
    
    node = await agent.get_node_by_id(kg, "REQ-001")
    
    assert node is not None
    assert node["id"] == "REQ-001"
    assert node["text"] == "Auth"


@pytest.mark.asyncio
async def test_get_edges_by_node(agent):
    """Testet das Abrufen aller Edges eines Nodes"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"},
            {"source": "REQ-002", "target": "REQ-001", "type": "CONFLICTS_WITH"}
        ]
    }
    
    edges = await agent.get_edges_by_node(kg, "REQ-001")
    
    assert len(edges) == 2
    assert edges[0]["source"] == "REQ-001"
    assert edges[1]["source"] == "REQ-002"


@pytest.mark.asyncio
async def test_get_dependencies(agent):
    """Testet das Abrufen aller Abhängigkeiten eines Nodes"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"},
            {"source": "REQ-001", "target": "REQ-002", "type": "CONFLICTS_WITH"}
        ]
    }
    
    dependencies = await agent.get_dependencies(kg, "REQ-001")
    
    assert len(dependencies) == 1
    assert dependencies[0]["type"] == "DEPENDS_ON"


@pytest.mark.asyncio
async def test_get_conflicts(agent):
    """Testet das Abrufen aller Konflikte eines Nodes"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"},
            {"source": "REQ-001", "target": "REQ-002", "type": "CONFLICTS_WITH"}
        ]
    }
    
    conflicts = await agent.get_conflicts(kg, "REQ-001")
    
    assert len(conflicts) == 1
    assert conflicts[0]["type"] == "CONFLICTS_WITH"


def test_parse_knowledge_graph(agent):
    """Testet das Parsen des Knowledge Graph aus der LLM-Antwort"""
    response = json.dumps({
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"}
        ]
    })
    
    kg = agent._parse_knowledge_graph(response)
    
    assert "nodes" in kg
    assert "edges" in kg
    assert len(kg["nodes"]) == 2
    assert len(kg["edges"]) == 1
