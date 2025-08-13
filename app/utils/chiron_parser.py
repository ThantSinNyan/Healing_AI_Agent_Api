import re
from app.models.healing_model import ChironAnalysisResponse

def parse_chiron_text(answer_text: str) -> ChironAnalysisResponse:
    """Parses raw Chiron analysis text into a ChironAnalysisResponse object."""

    def extract_section(name):
        # Match the heading and capture content until the next heading or end of text
        pattern = rf"\*\*{name}\*\*:([\s\S]*?)(?=\n\*\*|$)"
        match = re.search(pattern, answer_text)
        return match.group(1).strip() if match else ""

    placement = extract_section("Placement")
    core_wounded = extract_section("Core Wounded Themes")
    summary = extract_section("Summary Overview")
    wounded_keywords = extract_section("Wounded Keywords")
    healing_keywords = extract_section("Healing Keywords")
    primary_challenges = extract_section("Primary Challenges")
    path_to_healing = extract_section("Path to Healing")

    # Convert certain sections into lists
    summary_list = [s.strip() for s in summary.split("\n") if s.strip()]
    wounded_keywords_list = [w.strip() for w in re.split(r",|\n", wounded_keywords) if w.strip()]
    healing_keywords_list = [w.strip() for w in re.split(r",|\n", healing_keywords) if w.strip()]
    primary_challenges_list = [c.strip("• ").strip() for c in primary_challenges.split("\n") if c.strip()]
    path_to_healing_list = [p.strip("• ").strip() for p in path_to_healing.split("\n") if p.strip()]

    return ChironAnalysisResponse(
        placement=placement,
        coreWoundedThemes=core_wounded,
        summaryOverview=summary_list,
        woundedKeywords=wounded_keywords_list,
        healingKeywords=healing_keywords_list,
        primaryChallenges=primary_challenges_list,
        pathToHealing=path_to_healing_list
    )
