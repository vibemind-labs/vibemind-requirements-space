"""
Unit-Tests für RequirementsSummaryAgent
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from arch_team.agents.requirements_summary_agent import RequirementsSummaryAgent


@pytest.fixture
def agent():
    """Fixture für RequirementsSummaryAgent"""
    return RequirementsSummaryAgent()


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Testet die Agent-Initialisierung"""
    assert agent is not None
    assert agent.assistant is not None
    assert agent.user_proxy is not None
    assert agent.summary_tool is not None


@pytest.mark.asyncio
async def test_generate_summary(agent):
    """Testet die Zusammenfassungs-Generierung"""
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
        },
        {
            "id": "REQ-003",
            "text": "Admin users can reset user passwords.",
            "tag": "Functional",
            "priority": "Should",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "total_requirements": 3,
        "by_tag": {
            "Security": 2,
            "Functional": 1
        },
        "by_priority": {
            "Must": 2,
            "Should": 1
        },
        "conflicts": [],
        "dependencies": [],
        "quality_score": 100.0
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        summary = await agent.generate_summary(requirements)
        
        assert "total_requirements" in summary
        assert "by_tag" in summary
        assert "by_priority" in summary
        assert "conflicts" in summary
        assert "dependencies" in summary
        assert "quality_score" in summary
        assert summary["total_requirements"] == 3


@pytest.mark.asyncio
async def test_calculate_by_tag(agent):
    """Testet die Berechnung der Verteilung nach Tags"""
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
        },
        {
            "id": "REQ-003",
            "text": "Admin users can reset user passwords.",
            "tag": "Functional",
            "priority": "Should",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    by_tag = agent._calculate_by_tag(requirements)
    
    assert by_tag["Security"] == 2
    assert by_tag["Functional"] == 1


@pytest.mark.asyncio
async def test_calculate_by_priority(agent):
    """Testet die Berechnung der Verteilung nach Prioritäten"""
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
        },
        {
            "id": "REQ-003",
            "text": "Admin users can reset user passwords.",
            "tag": "Functional",
            "priority": "Should",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    by_priority = agent._calculate_by_priority(requirements)
    
    assert by_priority["Must"] == 2
    assert by_priority["Should"] == 1


@pytest.mark.asyncio
async def test_extract_conflicts(agent):
    """Testet die Extraktion von Konflikten"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "AES-256"},
            {"id": "REQ-002", "type": "Requirement", "text": "RSA-2048"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "CONFLICTS_WITH"}
        ]
    }
    
    conflicts = agent._extract_conflicts(kg)
    
    assert len(conflicts) == 1
    assert conflicts[0]["type"] == "CONFLICTS_WITH"


@pytest.mark.asyncio
async def test_extract_dependencies(agent):
    """Testet die Extraktion von Abhängigkeiten"""
    kg = {
        "nodes": [
            {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
            {"id": "REQ-002", "type": "Requirement", "text": "MFA"}
        ],
        "edges": [
            {"source": "REQ-001", "target": "REQ-002", "type": "DEPENDS_ON"}
        ]
    }
    
    dependencies = agent._extract_dependencies(kg)
    
    assert len(dependencies) == 1
    assert dependencies[0]["type"] == "DEPENDS_ON"


@pytest.mark.asyncio
async def test_calculate_quality_score(agent):
    """Testet die Berechnung des Quality-Scores"""
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"],
            "is_valid": True
        },
        {
            "id": "REQ-002",
            "text": "",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"],
            "is_valid": False
        }
    ]
    
    quality_score = agent._calculate_quality_score(requirements)
    
    assert quality_score == 50.0


def test_parse_summary(agent):
    """Testet das Parsen der Zusammenfassung aus der LLM-Antwort"""
    response = json.dumps({
        "total_requirements": 3,
        "by_tag": {
            "Security": 2,
            "Functional": 1
        },
        "by_priority": {
            "Must": 2,
            "Should": 1
        },
        "conflicts": [],
        "dependencies": [],
        "quality_score": 100.0
    })
    
    summary = agent._parse_summary(response)
    
    assert "total_requirements" in summary
    assert "by_tag" in summary
    assert "by_priority" in summary
    assert "conflicts" in summary
    assert "dependencies" in summary
    assert "quality_score" in summary
    assert summary["total_requirements"] == 3
