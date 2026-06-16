"""Turn module summaries into plain-language report blocks for the UI."""

VERDICT_LABELS = {
    "deploy": "Deploy",
    "deploy_with_restrictions": "Deploy with restrictions",
    "do_not_deploy": "Do not deploy",
    "inconclusive": "Inconclusive",
}


def metric_blocks(margin, direction):
    m = margin["summary"]
    d = direction["summary"]
    return [
        {
            "id": "safety_margin",
            "title": "Safety margin — behavior",
            "headline": f"fail_rate = {m['fail_rate']:.0%}",
            "fields": {
                "fail_rate": m["fail_rate"],
                "harm_vs_benign_auroc": m["harm_vs_benign_auroc"],
                "mean_margin_harmful": m["mean_margin_harmful"],
                "severity": m["severity"],
            },
            "what": (
                "Fraction of harmful prompts where complying is the cheaper path for the "
                "model. Computed by teacher-forcing canonical continuations: "
                "margin = logP(refuse) - logP(comply). This is the GCG attack objective, "
                "so it acts as a jailbreakability proxy."
            ),
            "read": (
                "Lower is safer. 0% means the model never preferred to comply on a harmful "
                "prompt; a high fail_rate means it is easy to steer into answering."
            ),
        },
        {
            "id": "refusal_direction",
            "title": "Refusal direction — harm awareness",
            "headline": f"AUROC = {d['separation_auroc']:.2f}",
            "fields": {
                "separation_auroc": d["separation_auroc"],
                "best_layer": d["best_layer"],
                "cohens_d": d["separation_cohens_d"],
                "severity": d["severity"],
            },
            "what": (
                "Difference-of-means 'refusal direction' in the residual stream "
                "(Arditi 2024). The AUROC measures how cleanly the internal state separates "
                "harmful from benign — a measure of harm awareness, not refusal itself."
            ),
            "read": (
                "Higher is better. ~1.0 means the model internally 'knows' a prompt is "
                "harmful; near 0.5 means no internal harm representation (typical of base "
                "models)."
            ),
        },
    ]


def build(repo, margin, direction, report, meta):
    v = report["summary"]
    return {
        "repo": repo,
        "verdict": {
            "code": v["verdict"],
            "label": VERDICT_LABELS.get(v["verdict"], v["verdict"]),
            "behavior": v["behavior"],
            "diagnosis": v["diagnosis"],
            "recommendation": v["recommendation"],
            "represents_harm": v.get("represents_harm"),
        },
        "metrics": metric_blocks(margin, direction),
        "meta": meta,
    }
