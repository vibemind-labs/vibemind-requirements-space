"""
Unit-Tests für MermaidGeneratorAgent
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from arch_team.agents.mermaid_generator_agent import MermaidGeneratorAgent


@pytest.fixture
def agent():
    """Fixture für MermaidGeneratorAgent"""
    return MermaidGeneratorAgent()


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Testet die Agent-Initialisierung"""
    assert agent is not None
    assert agent.assistant is not None
    assert agent.user_proxy is not None
    assert agent.generation_tool is not None


@pytest.mark.asyncio
async def test_generate_mermaid_diagram(agent):
    """Testet die Mermaid-Diagramm-Generierung"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"}
        ]
    }
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "mermaid": "graph TD\n  REQ001[REQ-001: Auth] --> REQ002[REQ-002: Encryption]",
        "diagram_type": "dependency",
        "metadata": {
            "nodes_count": 2,
            "edges_count": 1,
            "groups_count": 0
        }
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        mermaid_result = await agent.generate_mermaid_diagram(kg, "dependency")
        
        assert "mermaid" in mermaid_result
        assert "diagram_type" in mermaid_result
        assert "metadata" in mermaid_result
        assert "graph" in mermaid_result["mermaid"]


@pytest.mark.asyncio
async def test_validate_mermaid_syntax(agent):
    """Testet die Mermaid-Syntax-Validierung"""
    # Gültiges Mermaid-Diagramm
    valid_mermaid = "graph TD\n  REQ001[REQ-001: Auth] --> REQ002[REQ-002: Encryption]"
    result = await agent.validate_mermaid_syntax(valid_mermaid)
    
    assert result["valid"] == True
    assert len(result["issues"]) == 0
    
    # Ungültiges Mermaid-Diagramm
    invalid_mermaid = "invalid mermaid"
    result = await agent.validate_mermaid_syntax(invalid_mermaid)
    
    assert result["valid"] == False
    assert len(result["issues"]) > 0


@pytest.mark.asyncio
async def test_generate_dependency_diagram(agent):
    """Testet die Generierung eines Dependency-Diagramms"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"}
        ]
    }
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "mermaid": "graph TD\n  REQ001[REQ-001: Auth] --> REQ002[REQ-002: Encryption]",
        "diagram_type": "dependency",
        "metadata": {
            "nodes_count": 2,
            "edges_count": 1,
            "groups_count": 0
        }
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        mermaid_result = await agent.generate_dependency_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "dependency"
        assert "graph" in mermaid_result["mermaid"]


@pytest.mark.asyncio
async def test_generate_conflict_diagram(agent):
    """Testet die Generierung eines Conflict-Diagramms"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "AES-256"},
            {"id": "REQ-002", "type": "Requirement", "text": "RSA-2048"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "CONFLICTS_WITH"}
        ]
    }
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "mermaid": "graph LR\n  REQ001[REQ-001: AES-256] -.x. CONFLICTS_WITH .-> REQ002[REQ-002: RSA-2048]",
        "diagram_type": "conflict",
        "metadata": {
            "nodes_count": 2,
            "edges_count": 1,
            "groups_count": 0
        }
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        mermaid_result = await agent.generate_conflict_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "conflict"
        assert "graph" in mermaid_result["mermaid"]


@pytest.mark.asyncio
async def test_generate_grouped_diagram(agent):
    """Testet die Generierung eines Grouped-Diagramms"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "Encryption"},
            {"id": "Security", "type": "Tag", "text": "Security"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "Security", "type": "HAS_TAG"},
            {"source": "REQ-002", "target": "Security", "type": "HAS_TAG"}
        ]
    }
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "mermaid": "graph TD\n  subgraph Security\n    REQ001[REQ-001: Auth]\n    REQ002[REQ-002: Encryption]\n  end",
        "diagram_type": "grouped",
        "metadata": {
            "nodes_count": 3,
            "edges_count": 2,
            "groups_count": 1
        }
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        mermaid_result = await agent.generate_grouped_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "grouped"
        assert "subgraph" in mermaid_result["mermaid"]


@pytest.mark.asyncio
async def test_generate_actor_diagram(agent):
    """Testet die Generierung eines Actor-Diagramms"""
    kg = {
        "nodes": [
            {"id": "User", "type": "Actor", "text": "User"},
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"}
        ],
        "edges": [
            {"source": "User", "target": "REQ-001", "type": "ACTOR_RELATION"}
        ]
    }
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "mermaid": "graph LR\n  User[User] --> REQ001[REQ-001: Auth]",
        "diagram_type": "actor",
        "metadata": {
            "nodes_count": 2,
            "edges_count": 1,
            "groups_count": 0
        }
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        mermaid_result = await agent.generate_actor_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "actor"
        assert "graph" in mermaid_result["mermaid"]


def test_parse_mermaid_diagram(agent):
    """Testet das Parsen des Mermaid-Diagramms aus der LLM-Antwort"""
    response = json.dumps({
        "mermaid": "graph TD\n  REQ001[REQ-001: Auth] --> REQ002[REQ-002: Encryption]",
        "diagram_type": "dependency",
        "metadata": {
            "nodes_count": 2,
            "edges_count": 1,
            "groups_count": 0
        }
    })
    
    mermaid_result = agent._parse_mermaid_diagram(response)
    
    assert "mermaid" in mermaid_result
    assert "diagram_type" in mermaid_result
    assert "metadata" in mermaid_result
    assert mermaid_result["diagram_type"] == "dependency"
