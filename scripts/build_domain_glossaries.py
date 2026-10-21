"""Build reviewed, offline Wikidata label snapshots for subject terminology.

The QIDs below were resolved from the corresponding English Wikipedia topic
pages and reviewed as *concept IDs*. Labels remain community-maintained and
are not clinical, historical, or linguistic certification. The app never
contacts Wikidata at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_builtin_glossary import _read_entity  # noqa: E402

from pingyitong.language_test_results import VERIFIED_1_7B, VERIFIED_4B  # noqa: E402
from pingyitong.languages import LANGUAGES  # noqa: E402

# Deliberately avoid unqualified homonyms shared by different disciplines.
# The subject selector ensures a label such as "translation" in biology does
# not override an ordinary sentence in a different subject.
CONCEPTS: dict[str, dict[str, str]] = {
    "medicine": {
        "medical_diagnosis": "Q177719", "prognosis": "Q592442",
        "symptom": "Q169872", "infection": "Q166231",
        "inflammation": "Q101991", "vaccine": "Q134808",
        "immunization": "Q1415366", "antibody": "Q79460",
        "antibiotic": "Q12187", "pathogen": "Q170065",
        "surgery": "Q600236", "anesthesia": "Q131130",
        "radiography": "Q245341", "magnetic_resonance_imaging": "Q161238",
        "computed_tomography": "Q32566", "medical_ultrasound": "Q234904",
        "hypertension": "Q41861", "diabetes": "Q12206",
        "pneumonia": "Q12192", "cancer": "Q12078",
        "clinical_trial": "Q30612", "randomized_controlled_trial": "Q1436668",
        "blood_pressure": "Q82642", "heart_failure": "Q181754",
        "public_health": "Q189603", "epidemiology": "Q133805",
    },
    "biology": {
        "cell": "Q7868", "cell_membrane": "Q29548",
        "mitochondrion": "Q39572", "cell_nucleus": "Q40260",
        "dna": "Q7430", "rna": "Q11053", "gene": "Q7187",
        "chromosome": "Q37748", "genome": "Q7020",
        "protein": "Q8054", "enzyme": "Q8047",
        "amino_acid": "Q8066", "photosynthesis": "Q11982",
        "cellular_respiration": "Q912751", "evolution": "Q1063",
        "natural_selection": "Q43478", "species": "Q7432",
        "biodiversity": "Q47041", "ecosystem": "Q37813",
        "microbiome": "Q1330402", "bacteria": "Q10876",
        "virus": "Q808", "eukaryote": "Q19088",
        "prokaryote": "Q19081", "transcription": "Q177900",
        "translation": "Q185917", "mutation": "Q42918",
        "gene_expression": "Q26972",
    },
    "history": {
        "historiography": "Q50675", "archaeology": "Q23498",
        "archaeological_artifact": "Q220659", "dynasty": "Q164950",
        "empire": "Q48349", "feudalism": "Q37739",
        "renaissance": "Q4692", "industrial_revolution": "Q2269",
        "colonialism": "Q7167", "decolonization": "Q230533",
        "revolution": "Q10931", "chronology": "Q130788",
        "periodization": "Q816829", "historical_materialism": "Q190004",
        "oral_history": "Q558929", "historical_revisionism": "Q874803",
        "palaeography": "Q179957", "numismatics": "Q631286",
        "radiocarbon_dating": "Q173412", "ancient_history": "Q41493",
        "middle_ages": "Q12554", "modern_era": "Q3281534",
        "civilization": "Q8432", "primary_source": "Q112754",
    },
    "physics": {
        "thermodynamics": "Q11473", "electromagnetism": "Q11406",
        "quantum_mechanics": "Q944", "general_relativity": "Q11452",
        "special_relativity": "Q11455", "gravity": "Q11412",
        "inertia": "Q122508", "acceleration": "Q11376",
        "momentum": "Q41273", "kinetic_energy": "Q46276",
        "potential_energy": "Q155640", "electric_current": "Q11651",
        "magnetic_field": "Q11408", "wavelength": "Q41364",
        "frequency": "Q11652", "photon": "Q3198",
        "neutron": "Q2348", "proton": "Q2294",
        "atom": "Q9121", "entropy": "Q45003",
    },
    "chemistry": {
        "chemical_reaction": "Q36534", "molecule": "Q11369",
        "chemical_element": "Q11344", "chemical_compound": "Q11173",
        "covalent_bond": "Q127920", "ionic_bonding": "Q62500",
        "redox": "Q82682", "catalysis": "Q82264",
        "ph": "Q40936", "acid": "Q11158",
        "base": "Q11193", "solution": "Q5447188",
        "molar_concentration": "Q672821", "periodic_table": "Q10693",
        "isotope": "Q25276", "polymer": "Q81163",
        "organic_chemistry": "Q11351", "stoichiometry": "Q213185",
    },
    "mathematics": {
        "calculus": "Q149972", "derivative": "Q29175",
        "integral": "Q80091", "linear_algebra": "Q82571",
        "matrix": "Q44337", "euclidean_vector": "Q44528",
        "probability_theory": "Q5862903", "statistics": "Q12483",
        "theorem": "Q65943", "mathematical_proof": "Q11538",
        "equation": "Q11345", "function": "Q11348",
        "set_theory": "Q12482", "graph_theory": "Q131476",
        "differential_equation": "Q11214", "optimization": "Q141495",
        "logarithm": "Q11197", "geometry": "Q8087",
        "topology": "Q42989", "number_theory": "Q12479",
    },
}

LANGUAGE_TAGS = {
    "zh": ("zh-hans", "zh-cn", "zh"),
    "zh-TW": ("zh-hant", "zh-tw", "zh"),
    "jw": ("jv",),
    "no": ("nb", "nn", "no"),
}

# Some generic Wikidata Chinese labels are regional or traditional. These
# overrides follow common Simplified Chinese subject usage.
CURATED_ZH = {
    "medicine": {
        "medical_diagnosis": "医学诊断", "prognosis": "预后",
        "symptom": "症状", "infection": "感染", "inflammation": "炎症",
        "vaccine": "疫苗", "immunization": "免疫接种",
        "antibody": "抗体", "antibiotic": "抗生素", "pathogen": "病原体",
        "surgery": "外科手术", "anesthesia": "麻醉", "radiography": "X 射线摄影",
        "magnetic_resonance_imaging": "磁共振成像",
        "computed_tomography": "计算机断层扫描",
        "medical_ultrasound": "医学超声检查",
        "hypertension": "高血压", "diabetes": "糖尿病",
        "pneumonia": "肺炎", "cancer": "癌症",
        "clinical_trial": "临床试验", "randomized_controlled_trial": "随机对照试验",
        "blood_pressure": "血压", "heart_failure": "心力衰竭",
        "public_health": "公共卫生", "epidemiology": "流行病学",
    },
    "biology": {
        "cell": "细胞", "cell_membrane": "细胞膜", "mitochondrion": "线粒体",
        "cell_nucleus": "细胞核", "dna": "DNA", "rna": "RNA",
        "gene": "基因", "chromosome": "染色体", "genome": "基因组",
        "protein": "蛋白质", "enzyme": "酶", "amino_acid": "氨基酸",
        "photosynthesis": "光合作用", "cellular_respiration": "细胞呼吸",
        "evolution": "演化", "natural_selection": "自然选择",
        "species": "物种", "biodiversity": "生物多样性",
        "ecosystem": "生态系统", "microbiome": "微生物组",
        "bacteria": "细菌", "virus": "病毒", "eukaryote": "真核生物",
        "prokaryote": "原核生物", "transcription": "转录",
        "translation": "翻译", "mutation": "突变",
        "gene_expression": "基因表达",
    },
    "history": {
        "historiography": "历史编纂学", "archaeology": "考古学",
        "archaeological_artifact": "考古遗物", "dynasty": "王朝",
        "empire": "帝国", "feudalism": "封建制度",
        "renaissance": "文艺复兴", "industrial_revolution": "工业革命",
        "colonialism": "殖民主义", "decolonization": "去殖民化",
        "revolution": "革命", "chronology": "年代学",
        "periodization": "历史分期", "historical_materialism": "历史唯物主义",
        "oral_history": "口述历史", "historical_revisionism": "历史修正主义",
        "palaeography": "古文字学", "numismatics": "钱币学",
        "radiocarbon_dating": "放射性碳测年法",
        "ancient_history": "古代史", "middle_ages": "中世纪",
        "modern_era": "近代", "civilization": "文明",
        "primary_source": "一手史料",
    },
    "physics": {
        "thermodynamics": "热力学", "electromagnetism": "电磁学",
        "quantum_mechanics": "量子力学", "general_relativity": "广义相对论",
        "special_relativity": "狭义相对论", "gravity": "引力",
        "inertia": "惯性", "acceleration": "加速度", "momentum": "动量",
        "kinetic_energy": "动能", "potential_energy": "势能",
        "electric_current": "电流", "magnetic_field": "磁场",
        "wavelength": "波长", "frequency": "频率",
        "photon": "光子", "neutron": "中子", "proton": "质子",
        "atom": "原子", "entropy": "熵",
    },
    "chemistry": {
        "chemical_reaction": "化学反应", "molecule": "分子",
        "chemical_element": "化学元素", "chemical_compound": "化合物",
        "covalent_bond": "共价键", "ionic_bonding": "离子键",
        "redox": "氧化还原反应", "catalysis": "催化作用",
        "ph": "pH", "acid": "酸", "base": "碱",
        "solution": "溶液", "molar_concentration": "物质的量浓度",
        "periodic_table": "元素周期表", "isotope": "同位素",
        "polymer": "聚合物", "organic_chemistry": "有机化学",
        "stoichiometry": "化学计量学",
    },
    "mathematics": {
        "calculus": "微积分", "derivative": "导数", "integral": "积分",
        "linear_algebra": "线性代数", "matrix": "矩阵",
        "euclidean_vector": "向量", "probability_theory": "概率论",
        "statistics": "统计学", "theorem": "定理",
        "mathematical_proof": "数学证明", "equation": "方程",
        "function": "函数", "set_theory": "集合论",
        "graph_theory": "图论", "differential_equation": "微分方程",
        "optimization": "数学优化", "logarithm": "对数",
        "geometry": "几何学", "topology": "拓扑学",
        "number_theory": "数论",
    },
}

# Only unambiguous English variants are supplied. Other languages use the
# exact available Wikidata label rather than an inferred synonym.
CURATED_ALIASES = {
    "medicine": {
        "medical_diagnosis": {"en": ["diagnosis"]},
        "surgery": {"en": ["surgery"]},
        "anesthesia": {"en": ["anesthesia", "anaesthesia"]},
        "computed_tomography": {"en": ["CT scan"]},
        "magnetic_resonance_imaging": {"en": ["MRI"]},
        "medical_ultrasound": {"en": ["medical ultrasound"]},
        "hypertension": {"en": ["hypertension"]},
        "diabetes": {"en": ["diabetes mellitus"]},
    },
    "biology": {
        "cell_membrane": {"en": ["cell membrane"]},
        "dna": {"en": ["DNA"]},
        "rna": {"en": ["RNA"]},
        "prokaryote": {"en": ["prokaryote"]},
    },
    "history": {
        "archaeological_artifact": {"en": ["archaeological artifact"]},
        "palaeography": {"en": ["paleography"]},
    },
    "physics": {},
    "chemistry": {
        "molar_concentration": {"en": ["molar concentration", "molarity"]},
    },
    "mathematics": {
        "euclidean_vector": {"en": ["vector"]},
        "optimization": {"en": ["optimization"]},
    },
}


def build_domain(domain: str) -> dict:
    supported = sorted((VERIFIED_1_7B | VERIFIED_4B) & LANGUAGES.keys())
    concepts = []
    for slug, qid in CONCEPTS[domain].items():
        entity = _read_entity(qid)
        labels = entity["labels"]
        forms = {}
        for code in supported:
            for tag in LANGUAGE_TAGS.get(code, (code,)):
                label = labels.get(tag, {}).get("value", "").strip()
                if label:
                    forms[code] = label
                    break
        if not forms.get("en"):
            raise ValueError(f"{domain}/{slug}/{qid}: missing English label")
        forms.update({"zh": CURATED_ZH[domain][slug]})
        concepts.append({
            "id": slug,
            "wikidata_id": qid,
            "description_en": entity.get("descriptions", {}).get("en", {}).get("value", ""),
            "forms": forms,
            "aliases": CURATED_ALIASES[domain].get(slug, {}),
            "acronyms": [],
        })
        print(f"{domain}: {len(concepts)}/{len(CONCEPTS[domain])} {slug} = {forms['en']}", flush=True)
        time.sleep(0.3)
    return {
        "schema_version": 1,
        "domain": domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Wikidata entity labels (CC0) with reviewed Simplified Chinese forms",
        "source_url": "https://www.wikidata.org/wiki/Wikidata:Licensing",
        "supported_languages": supported,
        "invariant_terms": [],
        "concepts": concepts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "domains", nargs="*", metavar="DOMAIN",
        help="domains to rebuild (default: all new domains)",
    )
    parser.add_argument(
        "--reviewed-only", action="store_true",
        help="refresh reviewed Chinese forms and English aliases without network access",
    )
    args = parser.parse_args()
    unknown = set(args.domains) - CONCEPTS.keys()
    if unknown:
        parser.error(f"unknown domains: {', '.join(sorted(unknown))}")
    output_dir = ROOT / "config" / "glossaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains or CONCEPTS:
        output = output_dir / f"{domain}.json"
        if args.reviewed_only:
            data = json.loads(output.read_text(encoding="utf-8"))
            for concept in data["concepts"]:
                concept["forms"]["zh"] = CURATED_ZH[domain][concept["id"]]
                concept["aliases"] = CURATED_ALIASES[domain].get(concept["id"], {})
            output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"refreshed reviewed forms in {output}", flush=True)
            continue
        output.write_text(
            json.dumps(build_domain(domain), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
