"""Defense strategy generation based on attack analysis results."""

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# Defense recommendations per attack category
DEFENSE_STRATEGIES = {
    "dos": {
        "name": "Denial of Service (DoS)",
        "description": "Attacks aimed at making services unavailable by overwhelming them (slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed).",
        "strategies": [
            "Deploy rate limiting and traffic shaping on edge routers",
            "Configure SYN flood protection (SYN cookies, SYN proxy)",
            "Use application-layer DoS mitigation (mod_reqtimeout, mod_evasive)",
            "Implement slowloris-specific defenses (connection timeout tuning, max concurrent connections)",
            "Patch OpenSSL against Heartbleed (CVE-2014-0160) on all TLS endpoints",
            "Set up anomaly-based IDS to detect slow-rate attack patterns",
            "Deploy reverse proxies (Nginx, HAProxy) to absorb slow HTTP attacks",
        ],
        "priority": "HIGH",
        "monitoring": [
            "Monitor connection duration anomalies (long-lived idle connections)",
            "Set alerts for abnormal HTTP request rates and incomplete requests",
            "Track TLS heartbeat request frequency for Heartbleed detection",
        ],
    },
    "ddos": {
        "name": "Distributed Denial of Service (DDoS)",
        "description": "Large-scale volumetric attacks from multiple sources to exhaust bandwidth and resources.",
        "strategies": [
            "Use DDoS mitigation services (Cloudflare, AWS Shield, Akamai)",
            "Implement ingress/egress filtering (BCP38/BCP84)",
            "Deploy load balancers to distribute traffic across servers",
            "Configure rate limiting and connection throttling at edge",
            "Use anycast routing to absorb volumetric attacks",
            "Set up blackhole routing for targeted IP ranges during attacks",
            "Implement TCP SYN cookies and connection rate limiting",
        ],
        "priority": "CRITICAL",
        "monitoring": [
            "Monitor bandwidth utilization and packet rates in real-time",
            "Set alerts for traffic volume spikes from multiple sources",
            "Track SYN/ACK ratio anomalies and amplification patterns",
        ],
    },
    "brute_force": {
        "name": "Brute Force (FTP/SSH Patator)",
        "description": "Credential brute-force attacks against FTP and SSH services using dictionaries.",
        "strategies": [
            "Enforce strong password policies and multi-factor authentication",
            "Use SSH key-based authentication only (disable password auth)",
            "Deploy fail2ban or similar to block IPs after failed attempts",
            "Restrict SSH/FTP access to specific IP ranges via firewall",
            "Use non-standard ports for SSH and FTP services",
            "Implement account lockout after consecutive failures",
            "Deploy VPN for remote access instead of exposed SSH/FTP",
        ],
        "priority": "HIGH",
        "monitoring": [
            "Monitor failed login attempts and set lockout thresholds",
            "Alert on brute-force patterns (rapid sequential auth failures)",
            "Track login attempts from unusual geographic locations",
        ],
    },
    "web_attack": {
        "name": "Web Attacks (Brute Force, XSS, SQL Injection)",
        "description": "Application-layer attacks targeting web services including XSS, SQL injection, and web brute force.",
        "strategies": [
            "Deploy a Web Application Firewall (WAF) with OWASP rule sets",
            "Use parameterized queries / prepared statements to prevent SQL injection",
            "Implement Content Security Policy (CSP) headers against XSS",
            "Apply input validation and output encoding on all user inputs",
            "Enable HTTP-only and Secure flags on session cookies",
            "Use CAPTCHA or rate limiting on authentication endpoints",
            "Keep web frameworks and libraries patched and updated",
        ],
        "priority": "CRITICAL",
        "monitoring": [
            "Monitor WAF logs for blocked injection and XSS attempts",
            "Alert on unusual query patterns in database logs",
            "Track failed web authentication attempts by source IP",
        ],
    },
    "infiltration": {
        "name": "Infiltration",
        "description": "Stealthy attacks that establish a foothold inside the network using exploits or social engineering.",
        "strategies": [
            "Implement network segmentation with micro-segmentation",
            "Deploy endpoint detection and response (EDR) on all hosts",
            "Use least-privilege access control and zero-trust architecture",
            "Enable host-based firewalls and application whitelisting",
            "Monitor lateral movement with network traffic analysis",
            "Deploy honeypots and deception technology to detect intruders",
            "Regularly audit user accounts and access permissions",
        ],
        "priority": "CRITICAL",
        "monitoring": [
            "Monitor east-west traffic for anomalous internal connections",
            "Alert on new processes and unexpected outbound connections",
            "Track credential usage patterns for lateral movement indicators",
        ],
    },
    "botnet": {
        "name": "Botnet",
        "description": "Compromised hosts communicating with command-and-control servers for coordinated malicious activity.",
        "strategies": [
            "Deploy DNS sinkholing for known C2 domains",
            "Block outbound traffic to known malicious IPs (threat intelligence feeds)",
            "Monitor for periodic beaconing patterns in outbound traffic",
            "Use deep packet inspection to detect C2 protocols",
            "Implement egress filtering to restrict unauthorized outbound connections",
            "Deploy network behavior analysis to detect bot communication patterns",
            "Keep all systems patched to prevent initial compromise",
        ],
        "priority": "CRITICAL",
        "monitoring": [
            "Monitor for periodic outbound connections (beaconing)",
            "Alert on DNS queries to suspicious or newly registered domains",
            "Track outbound traffic to unusual ports and destinations",
        ],
    },
    "portscan": {
        "name": "Port Scanning / Reconnaissance",
        "description": "Network scanning to discover open ports, services, and vulnerabilities for future exploitation.",
        "strategies": [
            "Disable unnecessary services and close unused ports",
            "Configure firewall to detect and block port scanning patterns",
            "Deploy honeypots to detect and track reconnaissance activity",
            "Implement network segmentation to limit scan scope",
            "Use port knocking for sensitive services",
            "Enable logging on all network devices for scan detection",
            "Deploy NIDS with scan detection signatures (Snort/Suricata)",
        ],
        "priority": "MEDIUM",
        "monitoring": [
            "Alert on sequential port access patterns",
            "Monitor for ICMP sweep and TCP SYN scan activity",
            "Track failed connection attempts across multiple ports",
        ],
    },
    "benign": {
        "name": "Benign Traffic",
        "description": "Legitimate network traffic — no attack detected.",
        "strategies": [
            "Maintain baseline monitoring to detect future anomalies",
        ],
        "priority": "LOW",
        "monitoring": [
            "Continuously update traffic baseline profiles",
        ],
    },
}


def generate_strategies(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    results: list[dict],
) -> dict:
    """Generate defense strategies based on classification results.

    Returns:
        Dictionary with threat assessment and recommended strategies.
    """
    # Compute attack distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = {target_names[u]: int(c) for u, c in zip(unique, counts)}
    total = sum(pred_dist.values())

    # Find best model
    best = max(results, key=lambda r: r["f1_macro"])

    # Generate threat assessment
    threat_report = {
        "best_model": best["model"],
        "best_f1": best["f1_macro"],
        "best_accuracy": best["accuracy"],
        "detected_attacks": {},
        "defense_plan": [],
        "summary": "",
    }

    print(f"\n{'=' * 60}")
    print("  DEFENSE STRATEGY REPORT")
    print(f"{'=' * 60}")
    print(f"\n  Best model: {best['model']} (F1={best['f1_macro']:.4f})")
    print(f"\n  Detected attack distribution:")

    for category, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {category:10s}: {count:6d} ({pct:.1f}%)")

        if category != "normal":
            strat = DEFENSE_STRATEGIES.get(category, {})
            threat_report["detected_attacks"][category] = {
                "count": count,
                "percentage": round(pct, 1),
                "priority": strat.get("priority", "UNKNOWN"),
            }
            threat_report["defense_plan"].append({
                "category": category,
                "priority": strat.get("priority", "UNKNOWN"),
                "description": strat.get("description", ""),
                "strategies": strat.get("strategies", []),
                "monitoring": strat.get("monitoring", []),
            })

    # Sort defense plan by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    threat_report["defense_plan"].sort(
        key=lambda x: priority_order.get(x["priority"], 99)
    )

    # Print defense plan
    print(f"\n{'─' * 60}")
    print("  RECOMMENDED DEFENSE STRATEGIES")
    print(f"{'─' * 60}")

    for item in threat_report["defense_plan"]:
        cat = item["category"]
        print(f"\n  [{item['priority']}] {DEFENSE_STRATEGIES[cat]['name']}")
        print(f"  {item['description']}")
        print(f"  Strategies:")
        for s in item["strategies"]:
            print(f"    • {s}")
        print(f"  Monitoring:")
        for m in item["monitoring"]:
            print(f"    ◦ {m}")

    # Summary
    attack_count = total - pred_dist.get("normal", 0)
    attack_pct = attack_count / total * 100
    threat_report["summary"] = (
        f"Detected {attack_count} attacks ({attack_pct:.1f}% of traffic) "
        f"across {len(threat_report['detected_attacks'])} categories. "
        f"Best model: {best['model']} with F1={best['f1_macro']:.4f}."
    )

    print(f"\n{'─' * 60}")
    print(f"  SUMMARY: {threat_report['summary']}")
    print(f"{'=' * 60}\n")

    # --- Save reports to output/ ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Attack Analysis Report
    _save_attack_analysis(threat_report, pred_dist, total, best)

    # 2. Defense Strategies Report
    _save_defense_strategies(threat_report)

    return threat_report


def _save_attack_analysis(
    report: dict,
    pred_dist: dict,
    total: int,
    best: dict,
) -> None:
    """Save detailed attack analysis to output/attack_analysis_report.txt."""
    lines = []
    lines.append("=" * 60)
    lines.append("  ATTACK ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Model used:       {best['model']}")
    if best.get("f1_macro"):
        lines.append(f"  Model F1-score:   {best['f1_macro']:.4f}")
    if best.get("accuracy"):
        lines.append(f"  Model accuracy:   {best['accuracy']:.4f}")
    lines.append(f"  Total records:    {total}")

    attack_count = total - pred_dist.get("normal", 0)
    attack_pct = attack_count / total * 100
    lines.append(f"  Attacks detected: {attack_count} ({attack_pct:.1f}%)")
    lines.append(f"  Normal traffic:   {pred_dist.get('normal', 0)} ({100 - attack_pct:.1f}%)")

    lines.append("")
    lines.append("  DETECTED ATTACK DISTRIBUTION")
    lines.append("  " + "─" * 40)
    lines.append(f"  {'Category':<15} {'Count':>8} {'Percent':>8}  {'Priority':<10}")
    lines.append("  " + "─" * 40)

    for category, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        priority = DEFENSE_STRATEGIES.get(category, {}).get("priority", "—")
        lines.append(f"  {category:<15} {count:>8} {pct:>7.1f}%  {priority:<10}")

    lines.append("")
    lines.append("  THREAT ASSESSMENT PER CATEGORY")
    lines.append("  " + "─" * 40)

    for cat, info in sorted(
        report["detected_attacks"].items(),
        key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x[1]["priority"], 99),
    ):
        strat = DEFENSE_STRATEGIES.get(cat, {})
        lines.append("")
        lines.append(f"  [{info['priority']}] {strat.get('name', cat)}")
        lines.append(f"  Description: {strat.get('description', '')}")
        lines.append(f"  Detected:    {info['count']} records ({info['percentage']}%)")
        lines.append(f"  Risk level:  {info['priority']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  SUMMARY: {report['summary']}")
    lines.append("=" * 60)

    path = OUTPUT_DIR / "attack_analysis_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved: {path}")


def _save_defense_strategies(report: dict) -> None:
    """Save defense strategies and recommendations to output/defense_strategies.txt."""
    lines = []
    lines.append("=" * 60)
    lines.append("  DEFENSE STRATEGIES & RECOMMENDATIONS")
    lines.append("=" * 60)
    lines.append("")

    for item in report["defense_plan"]:
        cat = item["category"]
        strat = DEFENSE_STRATEGIES.get(cat, {})

        lines.append(f"  [{item['priority']}] {strat.get('name', cat)}")
        lines.append(f"  " + "─" * 50)
        lines.append(f"  {strat.get('description', '')}")
        lines.append("")

        lines.append("  Protection strategies:")
        for i, s in enumerate(item["strategies"], 1):
            lines.append(f"    {i}. {s}")
        lines.append("")

        lines.append("  Monitoring recommendations:")
        for i, m in enumerate(item["monitoring"], 1):
            lines.append(f"    {i}. {m}")
        lines.append("")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  IMPLEMENTATION PRIORITY ORDER")
    lines.append("=" * 60)
    lines.append("")

    priority_labels = {
        "CRITICAL": "Implement immediately — active exploitation risk",
        "HIGH": "Implement within 24-48 hours",
        "MEDIUM": "Implement within 1 week",
        "LOW": "Scheduled maintenance",
    }
    for item in report["defense_plan"]:
        cat = item["category"]
        strat = DEFENSE_STRATEGIES.get(cat, {})
        label = priority_labels.get(item["priority"], "")
        lines.append(f"  [{item['priority']}] {strat.get('name', cat)} — {label}")

    lines.append("")
    lines.append("=" * 60)

    path = OUTPUT_DIR / "defense_strategies.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved: {path}")
