#!/usr/bin/env python3
"""
Enhanced Reporting and Actionable Insights System

This module provides:
- Comprehensive analysis reporting with visualizations
- Actionable insights and recommendations
- Priority-based issue management
- Trend analysis and historical tracking
- Export capabilities (JSON, HTML, PDF)
- Interactive dashboards and summaries
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path

# Import analysis modules
try:
    from analysis import Issue, IssueCollection, IssueType, IssueCategory, IssueSeverity
    from performance_optimization import get_optimization_report
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

@dataclass
class ActionableInsight:
    """Represents an actionable insight with priority and impact."""
    title: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    impact: str    # "high", "medium", "low"
    effort: str    # "high", "medium", "low"
    category: str
    affected_files: List[str]
    recommended_actions: List[str]
    estimated_time: str  # e.g., "2 hours", "1 day"
    roi_score: float     # Return on investment score (0-10)

@dataclass
class TrendData:
    """Represents trend data for analysis metrics."""
    metric_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend_direction: str  # "improving", "degrading", "stable"
    time_period: str

class InsightGenerator:
    """
    Generate actionable insights from analysis results.
    
    Features:
    - Priority-based insight ranking
    - Impact and effort estimation
    - ROI calculation for fixes
    - Trend analysis
    """
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
    
    def generate_insights(self, issues: List[Issue], performance_data: Dict[str, Any] = None) -> List[ActionableInsight]:
        """Generate actionable insights from issues and performance data."""
        insights = []
        
        # Group issues by category and type
        issue_groups = self._group_issues(issues)
        
        # Generate insights for each category
        insights.extend(self._generate_security_insights(issue_groups))
        insights.extend(self._generate_performance_insights(issue_groups, performance_data))
        insights.extend(self._generate_quality_insights(issue_groups))
        insights.extend(self._generate_maintainability_insights(issue_groups))
        insights.extend(self._generate_dead_code_insights(issue_groups))
        
        # Sort by priority and ROI
        insights.sort(key=lambda x: (self._priority_score(x.priority), -x.roi_score), reverse=True)
        
        return insights
    
    def _group_issues(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group issues by category and type."""
        groups = defaultdict(lambda: defaultdict(list))
        
        for issue in issues:
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            issue_type = issue.type.value if hasattr(issue.type, 'value') else str(issue.type)
            groups[category][issue_type].append(issue)
        
        return groups
    
    def _generate_security_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate security-focused insights."""
        insights = []
        
        security_issues = issue_groups.get("security_vulnerability", {})
        
        if security_issues:
            total_security_issues = sum(len(issues) for issues in security_issues.values())
            
            # High-priority security insight
            if total_security_issues > 0:
                affected_files = list(set(
                    str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                    for issues in security_issues.values()
                    for issue in issues
                ))
                
                insights.append(ActionableInsight(
                    title="Critical Security Vulnerabilities Detected",
                    description=f"Found {total_security_issues} security vulnerabilities that could expose the application to attacks.",
                    priority="critical",
                    impact="high",
                    effort="medium",
                    category="security",
                    affected_files=affected_files,
                    recommended_actions=[
                        "Review and fix dangerous function usage (eval, exec, pickle.loads)",
                        "Implement input validation and sanitization",
                        "Replace unsafe subprocess calls with secure alternatives",
                        "Add security testing to CI/CD pipeline"
                    ],
                    estimated_time="1-2 days",
                    roi_score=9.5
                ))
        
        return insights
    
    def _generate_performance_insights(self, issue_groups: Dict, performance_data: Dict = None) -> List[ActionableInsight]:
        """Generate performance-focused insights."""
        insights = []
        
        performance_issues = issue_groups.get("performance_issue", {})
        
        if performance_issues:
            # Nested loops insight
            nested_loop_issues = performance_issues.get("inefficient_loop", [])
            if nested_loop_issues:
                insights.append(ActionableInsight(
                    title="Inefficient Nested Loops Detected",
                    description=f"Found {len(nested_loop_issues)} functions with nested loops causing O(n²) or higher complexity.",
                    priority="high",
                    impact="high",
                    effort="medium",
                    category="performance",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in nested_loop_issues
                    ],
                    recommended_actions=[
                        "Optimize algorithms to reduce time complexity",
                        "Use more efficient data structures (sets, dictionaries)",
                        "Consider caching or memoization for repeated calculations",
                        "Profile code to identify actual bottlenecks"
                    ],
                    estimated_time="4-8 hours",
                    roi_score=8.0
                ))
        
        # Performance monitoring insights
        if performance_data and "bottlenecks" in performance_data:
            bottlenecks = performance_data["bottlenecks"]
            if bottlenecks:
                insights.append(ActionableInsight(
                    title="Performance Bottlenecks Identified",
                    description=f"Analysis identified {len(bottlenecks)} performance bottlenecks in the codebase.",
                    priority="medium",
                    impact="medium",
                    effort="medium",
                    category="performance",
                    affected_files=[],
                    recommended_actions=[
                        "Optimize slow functions identified in performance report",
                        "Implement caching for frequently called functions",
                        "Consider code profiling for detailed analysis",
                        "Review algorithm efficiency"
                    ],
                    estimated_time="2-4 hours",
                    roi_score=7.0
                ))
        
        return insights
    
    def _generate_quality_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate code quality insights."""
        insights = []
        
        quality_issues = issue_groups.get("code_quality", {})
        
        if quality_issues:
            # High complexity insight
            complexity_issues = quality_issues.get("high_complexity", [])
            if complexity_issues:
                insights.append(ActionableInsight(
                    title="High Complexity Functions Need Refactoring",
                    description=f"Found {len(complexity_issues)} functions with high cyclomatic complexity that are difficult to maintain.",
                    priority="medium",
                    impact="medium",
                    effort="high",
                    category="maintainability",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in complexity_issues
                    ],
                    recommended_actions=[
                        "Break large functions into smaller, focused functions",
                        "Extract complex logic into separate methods",
                        "Reduce nesting depth using early returns",
                        "Add comprehensive unit tests before refactoring"
                    ],
                    estimated_time="1-2 days",
                    roi_score=6.5
                ))
            
            # Long lines insight
            long_line_issues = quality_issues.get("line_too_long", [])
            if long_line_issues:
                insights.append(ActionableInsight(
                    title="Code Formatting Issues Detected",
                    description=f"Found {len(long_line_issues)} lines that exceed the recommended length limit.",
                    priority="low",
                    impact="low",
                    effort="low",
                    category="formatting",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in long_line_issues
                    ],
                    recommended_actions=[
                        "Set up automatic code formatting (black, prettier)",
                        "Configure line length limits in IDE",
                        "Break long lines into multiple lines",
                        "Add pre-commit hooks for formatting"
                    ],
                    estimated_time="1-2 hours",
                    roi_score=4.0
                ))
        
        return insights
    
    def _generate_maintainability_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate maintainability insights."""
        insights = []
        
        # Check for incomplete implementations
        quality_issues = issue_groups.get("code_quality", {})
        incomplete_issues = quality_issues.get("incomplete_implementation", [])
        
        if incomplete_issues:
            insights.append(ActionableInsight(
                title="Incomplete Implementations Found",
                description=f"Found {len(incomplete_issues)} TODO/FIXME/HACK markers indicating incomplete work.",
                priority="medium",
                impact="medium",
                effort="medium",
                category="maintainability",
                affected_files=[
                    str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                    for issue in incomplete_issues
                ],
                recommended_actions=[
                    "Review and complete TODO items",
                    "Fix FIXME issues",
                    "Replace HACK solutions with proper implementations",
                    "Create tickets for items that can't be completed immediately"
                ],
                estimated_time="varies",
                roi_score=5.5
            ))
        
        return insights
    
    def _generate_dead_code_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate dead code insights."""
        insights = []
        
        dead_code_issues = issue_groups.get("dead_code", {})
        
        if dead_code_issues:
            total_dead_code = sum(len(issues) for issues in dead_code_issues.values())
            
            if total_dead_code > 5:  # Threshold for significant dead code
                insights.append(ActionableInsight(
                    title="Significant Dead Code Cleanup Needed",
                    description=f"Found {total_dead_code} unused code elements that can be safely removed.",
                    priority="low",
                    impact="medium",
                    effort="low",
                    category="cleanup",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issues in dead_code_issues.values()
                        for issue in issues
                    ],
                    recommended_actions=[
                        "Remove unused functions and classes",
                        "Clean up unused imports",
                        "Remove unused variables",
                        "Update documentation to reflect changes"
                    ],
                    estimated_time="2-4 hours",
                    roi_score=6.0
                ))
        
        return insights
    
    def _priority_score(self, priority: str) -> int:
        """Convert priority to numeric score for sorting."""
        scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return scores.get(priority, 0)
    
    def _load_insight_rules(self) -> Dict[str, Any]:
        """Load insight generation rules."""
        # This could be loaded from a configuration file
        return {
            "security_threshold": 1,
            "performance_threshold": 3,
            "complexity_threshold": 5,
            "dead_code_threshold": 5
        }

class TrendAnalyzer:
    """
    Analyze trends in codebase metrics over time.
    
    Features:
    - Historical data tracking
    - Trend detection
    - Regression identification
    - Progress monitoring
    """
    
    def __init__(self, history_file: str = "analysis_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def record_analysis(self, analysis_results: Dict[str, Any]):
        """Record analysis results for trend tracking."""
        timestamp = datetime.now().isoformat()
        
        # Extract key metrics
        metrics = {
            "timestamp": timestamp,
            "total_issues": analysis_results.get("summary", {}).get("total_issues", 0),
            "critical_issues": analysis_results.get("summary", {}).get("critical_issues", 0),
            "error_issues": analysis_results.get("summary", {}).get("error_issues", 0),
            "warning_issues": analysis_results.get("summary", {}).get("warning_issues", 0),
            "info_issues": analysis_results.get("summary", {}).get("info_issues", 0),
            "dead_code_items": analysis_results.get("summary", {}).get("dead_code_items", 0),
            "analysis_duration": analysis_results.get("duration", 0)
        }
        
        # Add issue type breakdown
        issue_types = analysis_results.get("issues", {}).get("by_type", {})
        metrics["issue_types"] = issue_types
        
        # Store in history
        self.history.append(metrics)
        
        # Keep only last 30 entries
        if len(self.history) > 30:
            self.history = self.history[-30:]
        
        # Save to file
        self._save_history()
    
    def get_trends(self) -> List[TrendData]:
        """Get trend analysis for key metrics."""
        trends = []
        
        if len(self.history) < 2:
            return trends
        
        current = self.history[-1]
        previous = self.history[-2]
        
        # Analyze trends for key metrics
        metrics_to_analyze = [
            "total_issues", "critical_issues", "error_issues", 
            "warning_issues", "dead_code_items", "analysis_duration"
        ]
        
        for metric in metrics_to_analyze:
            current_value = current.get(metric, 0)
            previous_value = previous.get(metric, 0)
            
            if previous_value > 0:
                change_percentage = ((current_value - previous_value) / previous_value) * 100
            else:
                change_percentage = 100 if current_value > 0 else 0
            
            # Determine trend direction
            if abs(change_percentage) < 5:
                trend_direction = "stable"
            elif change_percentage > 0:
                # For issues, increase is bad; for performance, depends on metric
                if metric == "analysis_duration":
                    trend_direction = "degrading"
                else:
                    trend_direction = "degrading"
            else:
                trend_direction = "improving"
            
            trends.append(TrendData(
                metric_name=metric,
                current_value=current_value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                trend_direction=trend_direction,
                time_period="since_last_analysis"
            ))
        
        return trends
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load analysis history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def _save_history(self):
        """Save analysis history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save analysis history: {e}")

class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    
    Features:
    - Multiple output formats (JSON, HTML, Markdown)
    - Interactive dashboards
    - Executive summaries
    - Detailed technical reports
    """
    
    def __init__(self):
        self.insight_generator = InsightGenerator()
        self.trend_analyzer = TrendAnalyzer()
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    issues: List[Issue] = None,
                                    performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        
        # Record analysis for trend tracking
        self.trend_analyzer.record_analysis(analysis_results)
        
        # Generate insights
        insights = []
        if issues:
            insights = self.insight_generator.generate_insights(issues, performance_data)
        
        # Get trends
        trends = self.trend_analyzer.get_trends()
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis_results, insights, trends)
        
        # Generate technical details
        technical_details = self._generate_technical_details(analysis_results, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights)
        
        # Compile comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "2.0",
                "analysis_engine": "comprehensive_codebase_analyzer"
            },
            "executive_summary": executive_summary,
            "actionable_insights": [asdict(insight) for insight in insights],
            "trend_analysis": [asdict(trend) for trend in trends],
            "technical_details": technical_details,
            "recommendations": recommendations,
            "raw_analysis": analysis_results
        }
        
        return report
    
    def _generate_executive_summary(self, analysis_results: Dict[str, Any], 
                                   insights: List[ActionableInsight],
                                   trends: List[TrendData]) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = analysis_results.get("summary", {})
        
        # Calculate health score (0-100)
        total_issues = summary.get("total_issues", 0)
        critical_issues = summary.get("critical_issues", 0)
        error_issues = summary.get("error_issues", 0)
        
        # Simple health score calculation
        if total_issues == 0:
            health_score = 100
        else:
            # Penalize critical and error issues more heavily
            penalty = (critical_issues * 10) + (error_issues * 5) + (total_issues * 1)
            health_score = max(0, 100 - penalty)
        
        # Determine overall status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "needs_attention"
        
        # Count high-priority insights
        high_priority_insights = len([i for i in insights if i.priority in ["critical", "high"]])
        
        # Analyze trends
        improving_trends = len([t for t in trends if t.trend_direction == "improving"])
        degrading_trends = len([t for t in trends if t.trend_direction == "degrading"])
        
        return {
            "health_score": health_score,
            "overall_status": status,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_priority_insights": high_priority_insights,
            "trend_summary": {
                "improving_metrics": improving_trends,
                "degrading_metrics": degrading_trends,
                "stable_metrics": len(trends) - improving_trends - degrading_trends
            },
            "key_findings": self._extract_key_findings(insights),
            "next_steps": self._generate_next_steps(insights)
        }
    
    def _generate_technical_details(self, analysis_results: Dict[str, Any], 
                                   issues: List[Issue] = None) -> Dict[str, Any]:
        """Generate technical details section."""
        details = {
            "analysis_metadata": {
                "duration": analysis_results.get("duration", 0),
                "timestamp": analysis_results.get("timestamp"),
                "repository": analysis_results.get("repository")
            },
            "issue_breakdown": analysis_results.get("issues", {}),
            "statistics": analysis_results.get("statistics", {}),
            "performance_metrics": analysis_results.get("performance", {})
        }
        
        # Add issue distribution analysis
        if issues:
            details["issue_distribution"] = self._analyze_issue_distribution(issues)
        
        return details
    
    def _generate_recommendations(self, insights: List[ActionableInsight]) -> Dict[str, Any]:
        """Generate prioritized recommendations."""
        # Group insights by priority
        by_priority = defaultdict(list)
        for insight in insights:
            by_priority[insight.priority].append(insight)
        
        # Generate immediate actions (critical/high priority)
        immediate_actions = []
        for insight in by_priority["critical"] + by_priority["high"]:
            immediate_actions.extend(insight.recommended_actions)
        
        # Generate short-term goals (medium priority)
        short_term_goals = []
        for insight in by_priority["medium"]:
            short_term_goals.extend(insight.recommended_actions)
        
        # Generate long-term improvements (low priority)
        long_term_improvements = []
        for insight in by_priority["low"]:
            long_term_improvements.extend(insight.recommended_actions)
        
        return {
            "immediate_actions": list(set(immediate_actions))[:5],  # Top 5 unique actions
            "short_term_goals": list(set(short_term_goals))[:5],
            "long_term_improvements": list(set(long_term_improvements))[:5],
            "estimated_effort": self._calculate_total_effort(insights),
            "expected_roi": self._calculate_expected_roi(insights)
        }
    
    def _extract_key_findings(self, insights: List[ActionableInsight]) -> List[str]:
        """Extract key findings from insights."""
        findings = []
        
        # Get top 3 highest priority insights
        top_insights = sorted(insights, key=lambda x: (self.insight_generator._priority_score(x.priority), -x.roi_score), reverse=True)[:3]
        
        for insight in top_insights:
            findings.append(f"{insight.title}: {insight.description}")
        
        return findings
    
    def _generate_next_steps(self, insights: List[ActionableInsight]) -> List[str]:
        """Generate next steps based on insights."""
        next_steps = []
        
        # Get critical and high priority insights
        priority_insights = [i for i in insights if i.priority in ["critical", "high"]]
        
        for insight in priority_insights[:3]:  # Top 3
            next_steps.append(f"Address {insight.title.lower()} (estimated time: {insight.estimated_time})")
        
        if not next_steps:
            next_steps.append("Continue monitoring code quality and address medium priority issues")
        
        return next_steps
    
    def _analyze_issue_distribution(self, issues: List[Issue]) -> Dict[str, Any]:
        """Analyze the distribution of issues."""
        # Count by category
        by_category = Counter()
        by_severity = Counter()
        by_file = Counter()
        
        for issue in issues:
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            
            by_category[category] += 1
            by_severity[severity] += 1
            
            if hasattr(issue, 'location') and issue.location:
                file_path = issue.location.file_path if hasattr(issue.location, 'file_path') else 'unknown'
                by_file[file_path] += 1
        
        return {
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "most_problematic_files": dict(by_file.most_common(10))
        }
    
    def _calculate_total_effort(self, insights: List[ActionableInsight]) -> str:
        """Calculate total estimated effort."""
        # Simple effort calculation
        effort_hours = 0
        for insight in insights:
            if "hour" in insight.estimated_time:
                hours = int(insight.estimated_time.split()[0].split('-')[0])
                effort_hours += hours
            elif "day" in insight.estimated_time:
                days = int(insight.estimated_time.split()[0].split('-')[0])
                effort_hours += days * 8  # 8 hours per day
        
        if effort_hours < 8:
            return f"{effort_hours} hours"
        else:
            return f"{effort_hours // 8} days"
    
    def _calculate_expected_roi(self, insights: List[ActionableInsight]) -> float:
        """Calculate expected return on investment."""
        if not insights:
            return 0.0
        
        total_roi = sum(insight.roi_score for insight in insights)
        return total_roi / len(insights)
    
    def export_html_report(self, report: Dict[str, Any], output_file: str = "analysis_report.html"):
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Codebase Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .insight {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .critical {{ border-left-color: #dc3545; }}
        .high {{ border-left-color: #fd7e14; }}
        .medium {{ border-left-color: #ffc107; }}
        .low {{ border-left-color: #28a745; }}
        .trend-improving {{ color: #28a745; }}
        .trend-degrading {{ color: #dc3545; }}
        .trend-stable {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Codebase Analysis Report</h1>
        <p>Generated: {report['metadata']['generated_at']}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Health Score:</strong> {report['executive_summary']['health_score']}/100</p>
        <p><strong>Overall Status:</strong> {report['executive_summary']['overall_status']}</p>
        <p><strong>Total Issues:</strong> {report['executive_summary']['total_issues']}</p>
        <p><strong>Critical Issues:</strong> {report['executive_summary']['critical_issues']}</p>
    </div>
    
    <h2>Actionable Insights</h2>
"""
        
        # Add insights
        for insight in report['actionable_insights']:
            priority_class = insight['priority']
            html += f"""
    <div class="insight {priority_class}">
        <h3>{insight['title']}</h3>
        <p>{insight['description']}</p>
        <p><strong>Priority:</strong> {insight['priority']} | <strong>Impact:</strong> {insight['impact']} | <strong>Effort:</strong> {insight['effort']}</p>
        <p><strong>Estimated Time:</strong> {insight['estimated_time']}</p>
        <ul>
"""
            for action in insight['recommended_actions']:
                html += f"<li>{action}</li>"
            
            html += "</ul></div>"
        
        html += """
    <h2>Recommendations</h2>
    <h3>Immediate Actions</h3>
    <ul>
"""
        
        for action in report['recommendations']['immediate_actions']:
            html += f"<li>{action}</li>"
        
        html += """
    </ul>
</body>
</html>
"""
        
        return html

def generate_enhanced_report(analysis_results: Dict[str, Any], 
                           issues: List[Issue] = None,
                           performance_data: Dict[str, Any] = None,
                           export_html: bool = False) -> Dict[str, Any]:
    """Generate enhanced analysis report with actionable insights."""
    
    generator = ReportGenerator()
    report = generator.generate_comprehensive_report(analysis_results, issues, performance_data)
    
    if export_html:
        html_file = generator.export_html_report(report)
        report['html_export'] = html_file
    
    return report

if __name__ == "__main__":
    # Example usage
    print("📊 Enhanced Reporting System")
    print("=" * 40)
    
    # This would typically be called with real analysis results
    sample_results = {
        "summary": {
            "total_issues": 15,
            "critical_issues": 2,
            "error_issues": 5,
            "warning_issues": 6,
            "info_issues": 2
        },
        "duration": 5.2,
        "timestamp": datetime.now().isoformat()
    }
    
    report = generate_enhanced_report(sample_results, export_html=True)
    print(f"Generated comprehensive report with {len(report['actionable_insights'])} insights")
    
    if 'html_export' in report:
        print(f"HTML report exported to: {report['html_export']}")

