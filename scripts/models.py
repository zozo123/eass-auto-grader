"""
Pydantic models for structured LLM grading output.

These models ensure consistent, validated JSON output from any LLM provider.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CrudCompleteness(str, Enum):
    """CRUD implementation completeness level."""
    full = "full"
    partial = "partial"
    minimal = "minimal"
    none = "none"


class Severity(str, Enum):
    """Issue severity level."""
    critical = "critical"
    major = "major"
    minor = "minor"
    suggestion = "suggestion"


class Issue(BaseModel):
    """An issue found in the project.
    
    Category is flexible to handle LLM variations like 'security' vs 'security_practices'.
    """
    severity: Severity
    category: str = Field(..., description="Issue category (e.g., security, code_quality, testing)")
    description: str


class Grade(str, Enum):
    """Letter grade."""
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"


class EvaluationMode(str, Enum):
    """Evaluation mode for the project."""
    backend = "backend"
    fullstack = "fullstack"
    mvp = "mvp"



# =============================================================================
# Score component models
# =============================================================================

class Scores(BaseModel):
    """Individual scoring components (0-10 scale)."""
    functional_correctness: float = Field(..., ge=0, le=10, description="CRUD works, correct status codes")
    architecture_design: float = Field(..., ge=0, le=10, description="Separation of concerns, repository pattern")
    code_quality: float = Field(..., ge=0, le=10, description="Type hints, docstrings, DRY")
    api_design: float = Field(..., ge=0, le=10, description="RESTful, validation, response models")
    error_handling: float = Field(..., ge=0, le=10, description="HTTP exceptions, edge cases")
    security_practices: float = Field(..., ge=0, le=10, description="No hardcoded secrets, env vars")
    test_quality: float = Field(..., ge=0, le=10, description="Pytest coverage, isolation, error cases")
    documentation: float = Field(..., ge=0, le=10, description="README quality, setup instructions")
    docker_containerization: float = Field(..., ge=0, le=10, description="Dockerfile exists, best practices")
    dependency_management: float = Field(..., ge=0, le=10, description="Requirements/pyproject, pinned versions")
    creativity_and_value: float = Field(default=5.0, ge=0, le=10, description="Originality, product value, not trivial")


# =============================================================================
# Deep analysis component models
# =============================================================================

class FunctionalCorrectnessAnalysis(BaseModel):
    """Detailed functional correctness analysis."""
    score: float = Field(..., ge=0, le=10)
    endpoints_found: int = Field(..., ge=0, description="Number of API endpoints")
    crud_completeness: CrudCompleteness
    would_run: bool = Field(..., description="Would the app run without errors?")
    evidence: str = Field(..., description="Code evidence with file:line references")


class ArchitectureDesignAnalysis(BaseModel):
    """Detailed architecture analysis."""
    score: float = Field(..., ge=0, le=10)
    pattern: str = Field(..., description="Design pattern used (e.g., repository, MVC)")
    separation_of_concerns: str = Field(..., description="good/partial/poor")
    evidence: str


class CodeQualityAnalysis(BaseModel):
    """Detailed code quality analysis."""
    score: float = Field(..., ge=0, le=10)
    type_hints: str = Field(..., description="full/partial/none")
    docstrings: str = Field(..., description="comprehensive/some/none")
    evidence: str


class ApiDesignAnalysis(BaseModel):
    """Detailed API design analysis."""
    score: float = Field(..., ge=0, le=10)
    restful: bool
    response_models: bool
    validation: str = Field(..., description="strong/basic/none")
    evidence: str


class ErrorHandlingAnalysis(BaseModel):
    """Detailed error handling analysis."""
    score: float = Field(..., ge=0, le=10)
    http_exceptions: bool
    edge_cases: str = Field(..., description="comprehensive/partial/none")
    evidence: str


class SecurityPracticesAnalysis(BaseModel):
    """Detailed security analysis."""
    score: float = Field(..., ge=0, le=10)
    secrets_exposed: bool
    env_vars_used: bool
    evidence: str


class TestQualityAnalysis(BaseModel):
    """Detailed test quality analysis."""
    score: float = Field(..., ge=0, le=10)
    test_count: int = Field(..., ge=0)
    test_isolation: bool
    error_cases_tested: bool
    evidence: str


class DocumentationAnalysis(BaseModel):
    """Detailed documentation analysis."""
    score: float = Field(..., ge=0, le=10)
    readme_quality: str = Field(..., description="comprehensive/adequate/minimal/none")
    readme_formatted: bool = Field(..., description="Is the README properly formatted with Markdown?")
    setup_instructions: bool
    evidence: str


class DockerAnalysis(BaseModel):
    """Detailed Docker containerization analysis."""
    score: float = Field(..., ge=0, le=10)
    dockerfile_exists: bool
    docker_compose: bool
    best_practices: str = Field(..., description="good/partial/poor")
    evidence: str


class DependencyManagementAnalysis(BaseModel):
    """Detailed dependency management analysis."""
    score: float = Field(..., ge=0, le=10)
    tool: str = Field(..., description="pip/uv/poetry/none")
    versions_pinned: bool
    evidence: str


class CreativityAndValueAnalysis(BaseModel):
    """Detailed analysis of creativity, product value, and originality."""
    score: float = Field(..., ge=0, le=10)
    originality_level: str = Field(default="standard", description="trivial|standard|creative|innovative")
    product_value: str = Field(default="medium", description="none|low|medium|high - would someone use this?")
    domain_complexity: str = Field(default="moderate", description="simple|moderate|complex")
    tutorial_copy: bool = Field(default=False, description="True if appears copied from tutorial/documentation")
    evidence: str = Field(default="", description="Specific examples supporting the assessment")


class DeepAnalysis(BaseModel):
    """Comprehensive deep analysis of all scoring components."""
    functional_correctness: FunctionalCorrectnessAnalysis
    architecture_design: ArchitectureDesignAnalysis
    code_quality: CodeQualityAnalysis
    api_design: ApiDesignAnalysis
    error_handling: ErrorHandlingAnalysis
    security_practices: SecurityPracticesAnalysis
    test_quality: TestQualityAnalysis
    documentation: DocumentationAnalysis
    docker_containerization: DockerAnalysis
    dependency_management: DependencyManagementAnalysis
    creativity_and_value: Optional[CreativityAndValueAnalysis] = None


# =============================================================================
# Supporting models
# =============================================================================

class TechStack(BaseModel):
    """Technology stack detected in the project."""
    framework: str = Field(..., description="FastAPI/Flask/other")
    frontend: str = Field(..., description="none/streamlit/react/other")
    database: str = Field(..., description="SQLite/PostgreSQL/MongoDB/none")
    orm: str = Field(..., description="SQLModel/SQLAlchemy/none")
    testing_framework: str = Field(..., description="pytest/unittest/none")


class FileInventory(BaseModel):
    """Files detected in the project."""
    has_readme: bool
    has_dockerfile: bool
    has_docker_compose: bool
    has_gitignore: bool
    has_requirements: bool
    has_pyproject: bool
    has_tests_dir: bool


class CodeMetrics(BaseModel):
    """Code metrics."""
    total_python_files: int = Field(..., ge=0)
    total_test_files: int = Field(..., ge=0)
    num_api_endpoints: int = Field(..., ge=0)


class PassFail(BaseModel):
    """Pass/fail determination."""
    passed: bool
    grade: Grade
    reason: str


class FrameworkCompliance(BaseModel):
    """Framework compliance check (FastAPI required)."""
    uses_fastapi: bool
    uses_flask: bool
    frontend_tech: str
    penalty_applied: bool
    penalty_reason: Optional[str] = None


# =============================================================================
# Main grading response model
# =============================================================================

class GradingResponse(BaseModel):
    """
    Complete grading response from the LLM.
    
    This is the structured output schema that ensures consistent,
    validated JSON from any LLM provider (Gemini, Codex, local).
    """
    repo_name: str = Field(..., description="Repository/project name")
    summary: str = Field(..., description="Brief project description")
    project_type: str = Field(..., description="backend API/fullstack/other")
    
    scores: Scores = Field(..., description="Individual component scores (0-10)")
    final_score: float = Field(..., ge=0, le=10, description="Weighted average score")
    
    deep_analysis: DeepAnalysis = Field(..., description="Detailed analysis per component")
    tech_stack: TechStack = Field(..., description="Detected technology stack")
    file_inventory: FileInventory = Field(..., description="File presence checks")
    code_metrics: CodeMetrics = Field(..., description="Code metrics")
    
    pass_fail: PassFail
    framework_compliance: FrameworkCompliance
    evaluation_mode: EvaluationMode = Field(..., description="Mode used for evaluation (backend/fullstack/mvp)")

    # Additional optional lists for issues, strengths, and improvements
    issues: list[Issue] = Field(default_factory=list, description="Issues found")
    strengths: list[str] = Field(default_factory=list, description="Project strengths")
    improvements: list[str] = Field(default_factory=list, description="Suggested improvements")

    @field_validator('final_score')
    def validate_final_score(cls, v: float) -> float:
        """Round final_score to 2 decimal places."""
        try:
            return round(float(v), 2)
        except Exception:
            return v


# =============================================================================
# JSON Schema export for LLM prompts
# =============================================================================

def get_json_schema() -> dict:
    """Get JSON schema for LLM structured output."""
    return GradingResponse.model_json_schema()


def get_json_schema_str() -> str:
    """Get JSON schema as formatted string for prompts."""
    import json
    return json.dumps(get_json_schema(), indent=2)


if __name__ == "__main__":
    # Print schema for reference
    print(get_json_schema_str())
