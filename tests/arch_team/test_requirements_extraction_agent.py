"""
Unit-Tests für RequirementsExtractionAgent
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from arch_team.agents.requirements_extraction_agent import RequirementsExtractionAgent


@pytest.fixture
def agent():
    """Fixture für RequirementsExtractionAgent"""
    return RequirementsExtractionAgent()


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Testet die Agent-Initialisierung"""
    assert agent is not None
    assert agent.assistant is not None
    assert agent.user_proxy is not None
    assert agent.extraction_tool is not None


@pytest.mark.asyncio
async def test_extract_requirements_from_text(agent):
    """Testet die Requirements-Extraktion aus Text"""
    text = "The system must authenticate users within 2 seconds."
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "requirements": [
            {
                "id": "REQ-001",
                "text": "The system must authenticate users within 2 seconds.",
                "tag": "Security",
                "priority": "Must",
                "evidence_refs": ["chunk-1"]
            }
        ]
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        requirements = await agent.extract_requirements(text)
        
        assert len(requirements) > 0
        assert requirements[0]["text"] == text


@pytest.mark.asyncio
async def test_cross_chunk_context(agent):
    """Testet den Cross-Chunk Kontext"""
    chunks = [
        "Chunk 1: The system must authenticate users within 2 seconds.",
        "Chunk 2: User passwords must be encrypted using AES-256.",
        "Chunk 3: The system shall support multi-factor authentication."
    ]
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "requirements": [
            {
                "id": "REQ-001",
                "text": "The system must authenticate users within 2 seconds.",
                "tag": "Security",
                "priority": "Must",
                "evidence_refs": ["chunk-2"]
            }
        ]
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        requirements = await agent.extract_requirements_with_context(chunks, neighbor_refs=True)
        
        assert len(requirements) > 0


@pytest.mark.asyncio
async def test_deduplication(agent):
    """Testet die Deduplizierung"""
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
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    deduplicated = await agent.deduplicate_requirements(requirements)
    
    assert len(deduplicated) == 1
    assert deduplicated[0]["id"] == "REQ-001"


@pytest.mark.asyncio
async def test_validation(agent):
    """Testet die Validierung"""
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
            "text": "",
            "tag": "Security",
            "priority": "Invalid",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    validated = await agent.validate_requirements(requirements)
    
    assert len(validated) == 2
    assert validated[0]["is_valid"] == True
    assert validated[1]["is_valid"] == False
    assert len(validated[1]["validation_issues"]) > 0


@pytest.mark.asyncio
async def test_extract_requirements_from_file(agent, tmp_path):
    """Testet die Requirements-Extraktion aus einer Datei"""
    # Test-Datei erstellen
    test_file = tmp_path / "test_requirements.txt"
    test_file.write_text("The system must authenticate users within 2 seconds.")
    
    # Mock für LLM-Antwort
    mock_response = json.dumps({
        "requirements": [
            {
                "id": "REQ-001",
                "text": "The system must authenticate users within 2 seconds.",
                "tag": "Security",
                "priority": "Must",
                "evidence_refs": ["file-1"]
            }
        ]
    })
    
    with patch.object(agent.user_proxy, 'a_initiate_chat', new_callable=AsyncMock(return_value=mock_response)):
        requirements = await agent.extract_requirements_from_file(str(test_file))
        
        assert len(requirements) > 0
        assert requirements[0]["text"] == "The system must authenticate users within 2 seconds."


def test_create_cross_chunk_context(agent):
    """Testet die Erstellung von Cross-Chunk Kontext"""
    chunks = [
        "Chunk 1",
        "Chunk 2",
        "Chunk 3"
    ]
    
    # Mit Nachbar-Chunks
    context = agent._create_cross_chunk_context(chunks, 1, neighbor_refs=True)
    assert "[Previous Chunk]" in context
    assert "[Current Chunk]" in context
    assert "[Next Chunk]" in context
    
    # Ohne Nachbar-Chunks
    context = agent._create_cross_chunk_context(chunks, 1, neighbor_refs=False)
    assert "[Previous Chunk]" not in context
    assert "[Current Chunk]" in context
    assert "[Next Chunk]" not in context


def test_parse_requirements(agent):
    """Testet das Parsen von Requirements aus der LLM-Antwort"""
    response = json.dumps({
        "requirements": [
            {
                "id": "REQ-001",
                "text": "The system must authenticate users within 2 seconds.",
                "tag": "Security",
                "priority": "Must",
                "evidence_refs": ["chunk-1"]
            }
        ]
    })
    
    requirements = agent._parse_requirements(response)
    
    assert len(requirements) == 1
    assert requirements[0]["id"] == "REQ-001"
    assert requirements[0]["text"] == "The system must authenticate users within 2 seconds."
