"""
Auto-generated GraphQL schema for arXiv SuperReview.
Expose deep hierarchical fields for each paper assessment.
"""
import json
import graphene
from pathlib import Path

DATA_DIR = Path(__file__).parent / "json"

def _load_all():
    data = {}
    if DATA_DIR.exists():
        for fp in DATA_DIR.glob("*.json"):
            with open(fp, "r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                except Exception:
                    continue
                key = obj.get("metadata", {}).get("arxiv_id", fp.stem)
                data[key] = obj
    return data

DATA = _load_all()

class Tag(graphene.ObjectType):
    name = graphene.String()
    value = graphene.String()

class Location(graphene.ObjectType):
    pages = graphene.List(graphene.Int)
    section = graphene.String()
    figure = graphene.String()
    table = graphene.String()

class ClaimRow(graphene.ObjectType):
    claim = graphene.String()
    location = graphene.Field(Location)
    evidence = graphene.String()
    strength = graphene.String()
    missing = graphene.String()

class ClaimsMatrix(graphene.ObjectType):
    columns = graphene.List(graphene.String)
    rows = graphene.List(ClaimRow)

class ThreatsToValidity(graphene.ObjectType):
    internal = graphene.String()
    external = graphene.String()
    construct = graphene.String()

class MethodsTheoretical(graphene.ObjectType):
    definitions = graphene.String()
    lemmas = graphene.String()
    proof_gaps = graphene.String()
    boundary_cases = graphene.String()
    counterexamples = graphene.String()

class MethodsComputational(graphene.ObjectType):
    splits = graphene.String()
    ablations = graphene.String()
    calibration = graphene.String()
    robustness = graphene.String()
    fairness_harms = graphene.String()

class MethodsQualMixed(graphene.ObjectType):
    sampling = graphene.String()
    coding_scheme = graphene.String()
    saturation = graphene.String()
    triangulation = graphene.String()
    reflexivity = graphene.String()

class MethodsBioHuman(graphene.ObjectType):
    approvals_consent = graphene.String()
    privacy = graphene.String()
    risk_mgmt = graphene.String()
    preregistration = graphene.String()

class MethodsEvidenceAudit(graphene.ObjectType):
    design_setup = graphene.String()
    statistical_validity = graphene.String()
    threats_to_validity = graphene.Field(ThreatsToValidity)
    reproducibility = graphene.String()
    if_theoretical = graphene.Field(MethodsTheoretical)
    if_computational_ml = graphene.Field(MethodsComputational)
    if_qualitative_mixed = graphene.Field(MethodsQualMixed)
    if_biomedical_human = graphene.Field(MethodsBioHuman)

class ScoreItem(graphene.ObjectType):
    dimension = graphene.String()
    score = graphene.Float()
    rationale = graphene.String()

class Readiness(graphene.ObjectType):
    status = graphene.String()
    justification = graphene.String()

class LimitationRisk(graphene.ObjectType):
    item = graphene.String()
    type = graphene.String()
    location = graphene.Field(Location)

class ActionableImprovement(graphene.ObjectType):
    priority = graphene.Int()
    change = graphene.String()
    where = graphene.String()
    rationale = graphene.String()
    expected_impact = graphene.String()

class Question(graphene.ObjectType):
    question = graphene.String()
    location = graphene.Field(Location)

class Note(graphene.ObjectType):
    page = graphene.Int()
    anchor = graphene.String()
    note = graphene.String()

class ExecutiveSummary(graphene.ObjectType):
    text = graphene.String()

class NoveltySignificance(graphene.ObjectType):
    what_is_new = graphene.String()
    missing_comparisons = graphene.List(graphene.String)

class ClarityOrganization(graphene.ObjectType):
    title_abstract_alignment = graphene.String()
    structure_flow = graphene.String()
    figures_tables_readability = graphene.String()
    terminology_consistency = graphene.String()
    ambiguities = graphene.String()

class PageInfo(graphene.ObjectType):
    page_no = graphene.Int()
    char_len = graphene.Int()
    sha256 = graphene.String()

class Section(graphene.ObjectType):
    title = graphene.String()
    page_start = graphene.Int()
    page_end = graphene.Int()
    children = graphene.List(lambda: Section)

class DocumentStructure(graphene.ObjectType):
    pages = graphene.List(PageInfo)
    sections = graphene.List(Section)

class PipelineInfo(graphene.ObjectType):
    planner_model = graphene.String()
    review_model = graphene.String()
    verifier_model = graphene.String()
    formatter_model = graphene.String()

class ProvenanceTimestamps(graphene.ObjectType):
    planned_at = graphene.String()
    reviewed_at = graphene.String()

class ProvenanceParams(graphene.ObjectType):
    temperature = graphene.Int()
    prompt_version = graphene.String()
    notes = graphene.String()

class Provenance(graphene.ObjectType):
    pipeline = graphene.Field(PipelineInfo)
    timestamps = graphene.Field(ProvenanceTimestamps)
    parameters = graphene.Field(ProvenanceParams)

class Metadata(graphene.ObjectType):
    arxiv_id = graphene.String()
    title = graphene.String()
    authors = graphene.List(graphene.String)
    published = graphene.String()
    categories = graphene.List(graphene.String)
    pdf_sha256 = graphene.String()
    license = graphene.String()
    msc_classes = graphene.List(graphene.String)
    acm_classes = graphene.List(graphene.String)

class Review(graphene.ObjectType):
    executive_summary = graphene.Field(ExecutiveSummary)
    claims_matrix = graphene.Field(ClaimsMatrix)
    methods_evidence_audit = graphene.Field(MethodsEvidenceAudit)
    novelty_significance = graphene.Field(NoveltySignificance)
    clarity_organization = graphene.Field(ClarityOrganization)
    limitations_risks = graphene.List(LimitationRisk)
    actionable_improvements = graphene.List(ActionableImprovement)
    quality_scorecard = graphene.List(ScoreItem)
    readiness_level = graphene.Field(Readiness)
    questions_to_authors = graphene.List(Question)
    line_page_anchored_notes = graphene.List(Note)

class Assessment(graphene.ObjectType):
    metadata = graphene.Field(Metadata)
    provenance = graphene.Field(Provenance)
    document_structure = graphene.Field(DocumentStructure)
    review = graphene.Field(Review)
    verification = graphene.JSONString()
    arweave = graphene.JSONString()

class Query(graphene.ObjectType):
    paper = graphene.Field(Assessment, arxiv_id=graphene.String(required=True))
    all_ids = graphene.List(graphene.String)

    def resolve_paper(self, info, arxiv_id):
        return DATA.get(arxiv_id)

    def resolve_all_ids(self, info):
        return list(DATA.keys())

schema = graphene.Schema(query=Query)
