"""
idle_checker.py — Production-grade AWS idle resource detector.

Usage:
    idle-checker --region ap-south-1 --services ec2 lambda s3 --output table
    idle-checker --region us-east-1 --services all --days 30 --output json
    idle-checker --region us-west-2 --services lambda ecr --output csv --out-file idle.csv
    idle-checker --region us-east-1 --services all --output table --verbose

Install as a CLI tool:
    pip install boto3 rich
    pip install -e .          # if using pyproject.toml entry_points

Requirements:
    pip install boto3 rich
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Callable

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ─────────────────────────── logging ──────────────────────────────────── #

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("aws-idle-scanner")

# ─────────────────────────── data model ───────────────────────────────── #

ALL_SERVICES = ("ec2", "lambda", "s3", "rds", "eks", "cloudfront", "ecr")


@dataclass
class IdleResource:
    service: str
    resource_id: str
    region: str
    reason: str
    extra: dict = field(default_factory=dict)

    def flat(self) -> dict:
        """Flat dict for CSV / tabular output."""
        return {
            "service": self.service,
            "resource_id": self.resource_id,
            "region": self.region,
            "reason": self.reason,
            **{f"extra.{k}": v for k, v in self.extra.items()},
        }


# ─────────────────────────── helpers ──────────────────────────────────── #

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cutoff(days: int) -> datetime:
    return utc_now() - timedelta(days=days)


def has_metrics(
    cw,
    namespace: str,
    metric: str,
    dimensions: list[dict],
    days: int,
    stat: str = "Sum",
) -> bool:
    """Return True if at least one non-zero datapoint exists in the window."""
    try:
        resp = cw.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=dimensions,
            StartTime=utc_now() - timedelta(days=days),
            EndTime=utc_now(),
            Period=86400,
            Statistics=[stat],
        )
        return any(dp[stat] > 0 for dp in resp.get("Datapoints", []))
    except ClientError as exc:
        log.warning("CloudWatch query failed (%s/%s): %s", namespace, metric, exc)
        return True  # fail-open: don't falsely flag as idle


def _paginate(client, method: str, result_key: str, **kwargs) -> list:
    """Generic boto3 paginator helper."""
    paginator = client.get_paginator(method)
    items: list = []
    for page in paginator.paginate(**kwargs):
        items.extend(page.get(result_key, []))
    return items


# ─────────────────────────── scanners ─────────────────────────────────── #

def scan_ec2(region: str, days: int) -> list[IdleResource]:
    ec2 = boto3.client("ec2", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    idle: list[IdleResource] = []

    # Only consider running instances
    instances = _paginate(
        ec2,
        "describe_instances",
        "Reservations",
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}],
    )

    for res in instances:
        for inst in res.get("Instances", []):
            iid = inst["InstanceId"]
            itype = inst.get("InstanceType", "unknown")
            name_tag = next(
                (t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), ""
            )

            active = has_metrics(
                cw,
                "AWS/EC2",
                "CPUUtilization",
                [{"Name": "InstanceId", "Value": iid}],
                days,
                stat="Average",
            )

            if not active:
                idle.append(
                    IdleResource(
                        service="ec2",
                        resource_id=iid,
                        region=region,
                        reason=f"No CPU utilization datapoints in last {days} days",
                        extra={"name": name_tag, "instance_type": itype},
                    )
                )

    return idle


def scan_lambda(region: str, days: int) -> list[IdleResource]:
    client = boto3.client("lambda", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    idle: list[IdleResource] = []

    functions = _paginate(client, "list_functions", "Functions")

    for fn in functions:
        name = fn["FunctionName"]
        runtime = fn.get("Runtime", "unknown")
        last_modified = fn.get("LastModified", "")

        active = has_metrics(
            cw,
            "AWS/Lambda",
            "Invocations",
            [{"Name": "FunctionName", "Value": name}],
            days,
        )

        if not active:
            idle.append(
                IdleResource(
                    service="lambda",
                    resource_id=name,
                    region=region,
                    reason=f"No invocations in last {days} days",
                    extra={"runtime": runtime, "last_modified": last_modified},
                )
            )

    return idle


def scan_s3(region: str, days: int) -> list[IdleResource]:
    """
    Uses CloudWatch Storage Lens / S3 request metrics where available,
    falling back to BucketSizeBytes and NumberOfObjects daily metrics
    from the AWS/S3 namespace (requires storage metrics to be enabled on each bucket).
    Buckets with zero bytes AND created before the window are flagged idle.
    """
    s3 = boto3.client("s3")
    cw = boto3.client("cloudwatch", region_name="us-east-1")  # S3 metrics live in us-east-1
    idle: list[IdleResource] = []

    try:
        buckets = s3.list_buckets().get("Buckets", [])
    except ClientError as exc:
        log.error("Cannot list S3 buckets: %s", exc)
        return idle

    threshold = cutoff(days)

    for b in buckets:
        name = b["Name"]
        created: datetime = b["CreationDate"]

        # Skip buckets newer than the lookback window
        if created >= threshold:
            log.debug("Skipping recently created bucket: %s", name)
            continue

        # Attempt CloudWatch size metric (only available if bucket-level metrics enabled)
        has_objects = has_metrics(
            cw,
            "AWS/S3",
            "BucketSizeBytes",
            [
                {"Name": "BucketName", "Value": name},
                {"Name": "StorageType", "Value": "StandardStorage"},
            ],
            days,
            stat="Average",
        )

        has_requests = has_metrics(
            cw,
            "AWS/S3",
            "AllRequests",
            [
                {"Name": "BucketName", "Value": name},
                {"Name": "FilterId", "Value": "EntireBucket"},
            ],
            days,
        )

        if not has_objects and not has_requests:
            # Supplement with a cheap HeadBucket + list check
            try:
                result = s3.list_objects_v2(Bucket=name, MaxKeys=1)
                object_count = result.get("KeyCount", 0)
            except ClientError:
                object_count = -1  # access denied — skip

            if object_count == 0:
                idle.append(
                    IdleResource(
                        service="s3",
                        resource_id=name,
                        region="global",
                        reason=f"Bucket is empty and older than {days} days with no recent requests",
                        extra={"created": created.isoformat()},
                    )
                )

    return idle


def scan_rds(region: str, days: int) -> list[IdleResource]:
    rds = boto3.client("rds", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    idle: list[IdleResource] = []

    # Paginate RDS instances
    db_instances = _paginate(rds, "describe_db_instances", "DBInstances")

    for db in db_instances:
        db_id = db["DBInstanceIdentifier"]
        engine = db.get("Engine", "unknown")
        db_class = db.get("DBInstanceClass", "unknown")
        status = db.get("DBInstanceStatus", "unknown")

        # Only evaluate available instances
        if status != "available":
            log.debug("Skipping RDS instance %s (status: %s)", db_id, status)
            continue

        conn_active = has_metrics(
            cw,
            "AWS/RDS",
            "DatabaseConnections",
            [{"Name": "DBInstanceIdentifier", "Value": db_id}],
            days,
        )

        read_active = has_metrics(
            cw,
            "AWS/RDS",
            "ReadIOPS",
            [{"Name": "DBInstanceIdentifier", "Value": db_id}],
            days,
        )

        if not conn_active and not read_active:
            idle.append(
                IdleResource(
                    service="rds",
                    resource_id=db_id,
                    region=region,
                    reason=f"No DB connections or read IOPS in last {days} days",
                    extra={"engine": engine, "instance_class": db_class},
                )
            )

    return idle


def scan_eks(region: str, days: int) -> list[IdleResource]:
    """
    EKS clusters don't emit a single 'activity' metric. We use:
      - cluster_failed_node_count (non-zero = something is wrong, not idle)
      - node_cpu_utilization via Container Insights (if enabled)
    Fallback: flag clusters with no Container Insights data at all.
    """
    eks = boto3.client("eks", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    idle: list[IdleResource] = []

    try:
        cluster_names: list[str] = []
        paginator = eks.get_paginator("list_clusters")
        for page in paginator.paginate():
            cluster_names.extend(page.get("clusters", []))
    except ClientError as exc:
        log.error("Cannot list EKS clusters: %s", exc)
        return idle

    for name in cluster_names:
        try:
            detail = eks.describe_cluster(name=name)["cluster"]
        except ClientError as exc:
            log.warning("Cannot describe EKS cluster %s: %s", name, exc)
            continue

        status = detail.get("status", "")
        created_at: datetime = detail.get("createdAt", utc_now())
        k8s_version = detail.get("version", "unknown")

        if status != "ACTIVE":
            log.debug("Skipping EKS cluster %s (status: %s)", name, status)
            continue

        if created_at >= cutoff(days):
            log.debug("Skipping recently created EKS cluster: %s", name)
            continue

        # Container Insights node CPU metric
        node_active = has_metrics(
            cw,
            "ContainerInsights",
            "node_cpu_utilization",
            [{"Name": "ClusterName", "Value": name}],
            days,
            stat="Average",
        )

        if not node_active:
            idle.append(
                IdleResource(
                    service="eks",
                    resource_id=name,
                    region=region,
                    reason=(
                        f"No Container Insights node CPU data in last {days} days "
                        "(Container Insights may be disabled or cluster has no nodes)"
                    ),
                    extra={"k8s_version": k8s_version, "created": created_at.isoformat()},
                )
            )

    return idle


def scan_cloudfront(region: str, days: int) -> list[IdleResource]:  # region unused but kept for uniform signature
    cf = boto3.client("cloudfront")
    cw = boto3.client("cloudwatch", region_name="us-east-1")  # CF metrics only in us-east-1
    idle: list[IdleResource] = []

    # Paginate CloudFront distributions via Marker
    marker: str | None = None
    while True:
        try:
            kwargs: dict = {"MaxItems": "100"}
            if marker:
                kwargs["Marker"] = marker
            resp = cf.list_distributions(**kwargs)
        except ClientError as exc:
            log.error("Cannot list CloudFront distributions: %s", exc)
            break

        dist_list = resp.get("DistributionList", {})
        items = dist_list.get("Items", [])

        for dist in items:
            dist_id = dist["Id"]
            domain = dist.get("DomainName", "")
            enabled = dist.get("Enabled", True)
            origins = ", ".join(
                o.get("DomainName", "") for o in dist.get("Origins", {}).get("Items", [])
            )

            if not enabled:
                idle.append(
                    IdleResource(
                        service="cloudfront",
                        resource_id=dist_id,
                        region="global",
                        reason="Distribution is disabled",
                        extra={"domain": domain, "origins": origins},
                    )
                )
                continue

            active = has_metrics(
                cw,
                "AWS/CloudFront",
                "Requests",
                [
                    {"Name": "DistributionId", "Value": dist_id},
                    {"Name": "Region", "Value": "Global"},
                ],
                days,
            )

            if not active:
                idle.append(
                    IdleResource(
                        service="cloudfront",
                        resource_id=dist_id,
                        region="global",
                        reason=f"No requests in last {days} days",
                        extra={"domain": domain, "origins": origins},
                    )
                )

        if dist_list.get("IsTruncated"):
            marker = dist_list.get("NextMarker")
        else:
            break

    return idle


def scan_ecr(region: str, days: int) -> list[IdleResource]:
    """
    A repo is idle when:
      - No image has been pushed in `days` days, AND
      - No image pull (via CloudWatch ECR API metrics) in `days` days.
    """
    ecr = boto3.client("ecr", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    idle: list[IdleResource] = []

    threshold = cutoff(days)

    try:
        repos = _paginate(ecr, "describe_repositories", "repositories")
    except ClientError as exc:
        log.error("Cannot list ECR repos: %s", exc)
        return idle

    for repo in repos:
        name = repo["repositoryName"]
        repo_uri = repo.get("repositoryUri", "")

        # Check for recent pushes
        try:
            images = _paginate(
                ecr, "describe_images", "imageDetails", repositoryName=name
            )
        except ClientError as exc:
            log.warning("Cannot describe images for ECR repo %s: %s", name, exc)
            continue

        recent_push = any(
            img.get("imagePushedAt", datetime.min.replace(tzinfo=timezone.utc)) >= threshold
            for img in images
        )

        if recent_push:
            continue

        # Check for recent pulls via CloudWatch ECR metrics
        pull_active = has_metrics(
            cw,
            "AWS/ECR",
            "SuccessfulPullCount",
            [{"Name": "RepositoryName", "Value": name}],
            days,
        )

        if not pull_active:
            image_count = len(images)
            latest_push = (
                max(
                    (img.get("imagePushedAt") for img in images if img.get("imagePushedAt")),
                    default=None,
                )
            )
            idle.append(
                IdleResource(
                    service="ecr",
                    resource_id=name,
                    region=region,
                    reason=f"No image pushes or pulls in last {days} days",
                    extra={
                        "repo_uri": repo_uri,
                        "image_count": str(image_count),
                        "latest_push": latest_push.isoformat() if latest_push else "never",
                    },
                )
            )

    return idle


# ─────────────────────────── runner ───────────────────────────────────── #

SCANNER_MAP: dict[str, Callable[..., list[IdleResource]]] = {
    "ec2": scan_ec2,
    "lambda": scan_lambda,
    "s3": scan_s3,
    "rds": scan_rds,
    "eks": scan_eks,
    "cloudfront": scan_cloudfront,
    "ecr": scan_ecr,
}


def run_scans(
    services: list[str],
    region: str,
    days: int,
    workers: int = 6,
) -> dict[str, list[IdleResource]]:
    results: dict[str, list[IdleResource]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(SCANNER_MAP[svc], region, days): svc
            for svc in services
        }
        for future in as_completed(futures):
            svc = futures[future]
            try:
                results[svc] = future.result()
                log.info("%-12s → %d idle resource(s) found", svc.upper(), len(results[svc]))
            except (ClientError, BotoCoreError) as exc:
                log.error("%-12s scan failed: %s", svc.upper(), exc)
                results[svc] = []

    return results


# ─────────────────────────── output formatters ────────────────────────── #

def fmt_text(results: dict[str, list[IdleResource]]) -> str:
    lines = ["\n" + "=" * 55, "  AWS IDLE RESOURCE REPORT", "=" * 55]
    total = 0
    for svc, items in sorted(results.items()):
        lines.append(f"\n▸ {svc.upper()} ({len(items)} idle)")
        if not items:
            lines.append("    ✅  None found")
        else:
            for r in items:
                lines.append(f"    ⚠️  {r.resource_id}")
                lines.append(f"       Reason : {r.reason}")
                lines.append(f"       Region : {r.region}")
                for k, v in r.extra.items():
                    lines.append(f"       {k:<10}: {v}")
        total += len(items)
    lines += ["", "=" * 55, f"  Total idle resources: {total}", "=" * 55, ""]
    return "\n".join(lines)


def fmt_json(results: dict[str, list[IdleResource]]) -> str:
    payload = {
        "generated_at": utc_now().isoformat(),
        "services": {svc: [asdict(r) for r in items] for svc, items in results.items()},
        "summary": {svc: len(items) for svc, items in results.items()},
        "total_idle": sum(len(v) for v in results.values()),
    }
    return json.dumps(payload, indent=2, default=str)


def fmt_csv(results: dict[str, list[IdleResource]]) -> str:
    all_resources = [r for items in results.values() for r in items]
    if not all_resources:
        return "service,resource_id,region,reason\n"

    # Collect all possible extra keys to build a stable header
    extra_keys: list[str] = []
    for r in all_resources:
        for k in r.extra:
            col = f"extra.{k}"
            if col not in extra_keys:
                extra_keys.append(col)

    fieldnames = ["service", "resource_id", "region", "reason"] + extra_keys

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in all_resources:
        writer.writerow(r.flat())

    return buf.getvalue()


OUTPUT_FMT = {"text": fmt_text, "json": fmt_json, "csv": fmt_csv}

# ─────────────────────────── service metadata ─────────────────────────── #

SERVICE_ICONS = {
    "ec2":         "🖥 ",
    "lambda":      "λ ",
    "s3":          "🪣",
    "rds":         "🗄 ",
    "eks":         "☸ ",
    "cloudfront":  "🌐",
    "ecr":         "📦",
}

SERVICE_COLORS = {
    "ec2":        "bright_yellow",
    "lambda":     "bright_cyan",
    "s3":         "green",
    "rds":        "bright_blue",
    "eks":        "magenta",
    "cloudfront": "bright_white",
    "ecr":        "orange3",
}

# ─────────────────────────── rich table output ────────────────────────── #

def fmt_table(results: dict[str, list[IdleResource]]) -> str:
    """Renders nothing — the rich console output is printed directly by print_table()."""
    return ""   # sentinel; actual rendering handled in main() for table mode


def print_table(
    results: dict[str, list[IdleResource]],
    region: str,
    days: int,
    console: "Console",
) -> None:
    """Print a full rich terminal report to *console*."""
    total_idle = sum(len(v) for v in results.values())
    total_scanned = len(results)
    now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── header banner ──────────────────────────────────────────────────── #
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text("☁  AWS Idle Resource Checker", style="bold white"),
        Text(now_str, style="dim"),
    )
    header.add_row(
        Text(f"   Region: {region}   ·   Lookback: {days} days", style="dim cyan"),
        Text(
            f"{'⚠  ' + str(total_idle) + ' idle found' if total_idle else '✅  All clear'}",
            style="bold red" if total_idle else "bold green",
        ),
    )
    console.print(Panel(header, border_style="bright_black", padding=(0, 1)))
    console.print()

    # ── per-service tables ─────────────────────────────────────────────── #
    for svc, items in sorted(results.items()):
        icon  = SERVICE_ICONS.get(svc, "▸ ")
        color = SERVICE_COLORS.get(svc, "white")
        count = len(items)

        # section title
        status_badge = (
            Text(f" {count} idle ", style=f"bold black on red")
            if count else
            Text(" ✅ clean ", style="bold black on green")
        )
        svc_label = Text(f" {icon} {svc.upper()} ", style=f"bold {color}")
        title_row = Text.assemble(svc_label, " ", status_badge)
        console.print(title_row)

        if not items:
            console.print("   No idle resources found.\n", style="dim")
            continue

        # determine which extra columns exist for this service
        extra_keys: list[str] = []
        for r in items:
            for k in r.extra:
                if k not in extra_keys:
                    extra_keys.append(k)

        tbl = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=f"bold {color}",
            border_style="bright_black",
            expand=False,
            pad_edge=True,
            show_edge=True,
        )

        tbl.add_column("Resource ID",   style="bold white",       no_wrap=True,  min_width=24)
        tbl.add_column("Region",        style="dim cyan",          no_wrap=True,  min_width=14)
        tbl.add_column("Reason",        style="yellow",            no_wrap=False, min_width=36, max_width=60)
        for k in extra_keys:
            tbl.add_column(k.replace("_", " ").title(), style="dim", no_wrap=False, max_width=28)

        for r in items:
            row = [r.resource_id, r.region, r.reason]
            for k in extra_keys:
                row.append(str(r.extra.get(k, "")))
            tbl.add_row(*row)

        console.print(tbl)
        console.print()

    # ── summary footer ─────────────────────────────────────────────────── #
    summary = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_black",
        border_style="bright_black",
        expand=False,
    )
    summary.add_column("Service",      style="bold white",   min_width=12)
    summary.add_column("Idle Count",   style="bold",         min_width=10, justify="center")
    summary.add_column("Status",       min_width=10,         justify="center")

    for svc in sorted(results):
        count = len(results[svc])
        icon  = SERVICE_ICONS.get(svc, "")
        count_style = "bold red" if count else "dim green"
        status_text = Text("⚠  idle", style="red") if count else Text("✅ clean", style="green")
        summary.add_row(
            f"{icon} {svc.upper()}",
            Text(str(count), style=count_style),
            status_text,
        )

    console.print(Panel(
        summary,
        title="[bold]Summary[/bold]",
        title_align="left",
        border_style="bright_black",
        padding=(0, 1),
    ))

# ─────────────────────────── live-progress runner ─────────────────────── #

def run_scans_live(
    services: list[str],
    region: str,
    days: int,
    workers: int,
    console: "Console",
) -> dict[str, list[IdleResource]]:
    """Same as run_scans() but renders a rich live progress bar while scanning."""
    results: dict[str, list[IdleResource]] = {}
    task_ids: dict[str, TaskID] = {}

    progress = Progress(
        SpinnerColumn(spinner_name="dots2", style="bold cyan"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28, style="bright_black", complete_style="cyan"),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        for svc in services:
            icon = SERVICE_ICONS.get(svc, "")
            task_ids[svc] = progress.add_task(
                f"{icon} {svc.upper():<12}",
                total=1,
                status="scanning…",
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(SCANNER_MAP[svc], region, days): svc
                for svc in services
            }
            for future in as_completed(futures):
                svc = futures[future]
                tid = task_ids[svc]
                try:
                    found = future.result()
                    results[svc] = found
                    n = len(found)
                    status = f"[red]{n} idle[/red]" if n else "[green]clean[/green]"
                    progress.update(tid, advance=1, status=status)
                    log.debug("%-12s → %d idle resource(s)", svc.upper(), n)
                except (ClientError, BotoCoreError) as exc:
                    results[svc] = []
                    progress.update(tid, advance=1, status="[red]error[/red]")
                    log.error("%-12s scan failed: %s", svc.upper(), exc)

    console.print()
    return results


# ─────────────────────────── CLI ──────────────────────────────────────── #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idle-checker",
        description="Detect idle AWS resources across common services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  idle-checker --region ap-south-1 --services ec2 lambda s3 rds eks cloudfront ecr --output table
  idle-checker --region us-east-1  --services all --days 30 --output json
  idle-checker --region us-west-2  --services lambda ecr --output csv --out-file idle.csv
  idle-checker --region us-east-1  --services all --output table --verbose
""",
    )

    parser.add_argument("--region", required=True, help="AWS region to scan (e.g. ap-south-1)")
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        metavar="N",
        help="Lookback window in days (default: 60)",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        required=True,
        metavar="SVC",
        help=f"Services to scan. Use 'all' or any of: {', '.join(ALL_SERVICES)}",
    )
    parser.add_argument(
        "--output",
        choices=["table", "text", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--out-file",
        metavar="PATH",
        help="Write output to a file instead of stdout (not supported for --output table)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        metavar="N",
        help="Parallel scan threads (default: 6)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def resolve_services(raw: list[str]) -> list[str]:
    if "all" in raw:
        return list(ALL_SERVICES)
    unknown = set(raw) - set(ALL_SERVICES)
    if unknown:
        raise ValueError(f"Unknown service(s): {', '.join(sorted(unknown))}. Valid: {', '.join(ALL_SERVICES)}")
    return list(dict.fromkeys(raw))  # deduplicate while preserving order


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        services = resolve_services(args.services)
    except ValueError as exc:
        parser.error(str(exc))

    use_table = args.output == "table"

    if use_table and not HAS_RICH:
        parser.error(
            "--output table requires the 'rich' library.\n"
            "Install it with:  pip install rich"
        )

    if use_table:
        console = Console(stderr=False)
        console.print()
        console.print(
            f"[bold cyan]idle-checker[/bold cyan]  "
            f"[dim]scanning {len(services)} service(s) · region=[cyan]{args.region}[/cyan] · "
            f"lookback=[cyan]{args.days}d[/cyan][/dim]"
        )
        console.print()
        results = run_scans_live(services, args.region, args.days, args.workers, console)
        print_table(results, args.region, args.days, console)
        return

    # ── non-table modes ────────────────────────────────────────────────── #
    log.info(
        "Scanning %d service(s) in region=%s over last %d day(s): %s",
        len(services), args.region, args.days, ", ".join(services),
    )
    results = run_scans(services, args.region, args.days, workers=args.workers)

    formatter = OUTPUT_FMT[args.output]
    output = formatter(results)

    if args.out_file:
        try:
            with open(args.out_file, "w", encoding="utf-8") as fh:
                fh.write(output)
            log.info("Report written to %s", args.out_file)
        except OSError as exc:
            log.error("Could not write to %s: %s", args.out_file, exc)
            sys.exit(1)
    else:
        print(output)


if __name__ == "__main__":
    main()