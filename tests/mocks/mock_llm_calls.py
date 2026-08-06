"""
Mock-LLM-Calls für Tests
"""

from typing import Dict, Any, List


# Mock-LLM-Calls für Requirements-Extraktion
mock_llm_calls: Dict[str, Any] = {
    "requirements_extraction": {
        "prompt": "Extract requirements from the following text",
        "response": {
            "requirements": [
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
                    "text": "The system shall support multi-factor authentication.",
                    "tag": "Security",
                    "priority": "Must",
                    "evidence_refs": ["chunk-1"]
                },
                {
                    "id": "REQ-004",
                    "text": "Admin users can reset user passwords.",
                    "tag": "Functional",
                    "priority": "Should",
                    "evidence_refs": ["chunk-1"]
                },
                {
                    "id": "REQ-005",
                    "text": "The API must respond within 500 milliseconds.",
                    "tag": "Performance",
                    "priority": "Must",
                    "evidence_refs": ["chunk-1"]
                }
            ]
        }
    },
    "kg_relationship": {
        "prompt": "Extract relationships between requirements",
        "response": {
            "nodes": [
                {"id": "REQ-001", "type": "Requirement", "text": "Auth"},
                {"id": "REQ-002", "type": "Requirement", "text": "Encryption"},
                {"id": "REQ-003", "type": "Requirement", "text": "MFA"},
                {"id": "REQ-004", "type": "Requirement", "text": "Password Reset"},
                {"id": "REQ-005", "type": "Requirement", "text": "API Latency"},
                {"id": "User", "type": "Actor", "text": "User"},
                {"id": "Admin", "type": "Actor", "text": "Admin"},
                {"id": "Security", "type": "Tag", "text": "Security"},
                {"id": "Functional", "type": "Tag", "text": "Functional"},
                {"id": "Performance", "type": "Tag", "text": "Performance"}
            ],
            "edges": [
                {"source": "REQ-001", "target": "Security", "type": "HAS_TAG"},
                {"source": "REQ-002", "target": "Security", "type": "HAS_TAG"},
                {"source": "REQ-003", "target": "Security", "type": "HAS_TAG"},
                {"source": "REQ-004", "target": "Functional", "type": "HAS_TAG"},
                {"source": "REQ-005", "target": "Performance", "type": "HAS_TAG"},
                {"source": "User", "target": "REQ-001", "type": "ACTOR_RELATION"},
                {"source": "User", "target": "REQ-002", "type": "ACTOR_RELATION"},
                {"source": "Admin", "target": "REQ-004", "type": "ACTOR_RELATION"},
                {"source": "REQ-001", "target": "REQ-003", "type": "DEPENDS_ON"}
            ]
        }
    },
    "mermaid_generator": {
        "prompt": "Generate Mermaid diagram from knowledge graph",
        "response": {
            "mermaid": """graph TD
    subgraph Security
        REQ001[REQ-001: Auth]
        REQ002[REQ-002: Encryption]
        REQ003[REQ-003: MFA]
    end
    subgraph Functional
        REQ004[REQ-004: Password Reset]
    end
    subgraph Performance
        REQ005[REQ-005: API Latency]
    end
    
    User --> REQ001
    User --> REQ002
    Admin --> REQ004
    REQ001 --> REQ003"""
        }
    },
    "requirements_summary": {
        "prompt": "Generate summary of requirements",
        "response": {
            "total_requirements": 5,
            "by_tag": {
                "functional": 1,
                "security": 3,
                "performance": 1
            },
            "by_priority": {
                "must": 4,
                "should": 1,
                "may": 0
            },
            "conflicts": [],
            "dependencies": [
                {"source": "REQ-001", "target": "REQ-003", "type": "DEPENDS_ON"}
            ]
        }
    },
    "output_validator": {
        "prompt": "Validate output",
        "response": {
            "syntax_valid": True,
            "consistency_valid": True,
            "completeness_valid": True,
            "semantic_valid": True,
            "cross_reference_valid": True,
            "issues": []
        }
    }
}


def get_mock_llm_call(call_type: str) -> Dict[str, Any]:
    """
    Gibt einen Mock-LLM-Call zurück
    
    Args:
        call_type (str): Typ des Mock-LLM-Calls
    
    Returns:
        Dict[str, Any]: Mock-LLM-Call
    """
    if call_type not in mock_llm_calls:
        raise ValueError(f"Mock-LLM-Call {call_type} ist nicht definiert")
    
    return mock_llm_calls[call_type]


def get_mock_llm_response(call_type: str) -> Any:
    """
    Gibt eine Mock-LLM-Antwort zurück
    
    Args:
        call_type (str): Typ des Mock-LLM-Calls
    
    Returns:
        Any: Mock-LLM-Antwort
    """
    mock_call = get_mock_llm_call(call_type)
    return mock_call["response"]
