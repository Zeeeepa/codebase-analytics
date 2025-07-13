"""
Advanced Reporting Engine
Generates comprehensive ERROR/CODEBASE-STATE reports
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.graph_manager import GraphSitterManager, CodeError, ErrorSeverity, ErrorCategory, FunctionImportance
from ..detectors.error_engine import AdvancedErrorDetector


@dataclass
class CodebaseHealthMetrics:
    """Comprehensive codebase health metrics"""
    maintainability_score: float  # 0-100
    technical_debt_ratio: float   # 0-1
    error_density: float          # errors per 1000 lines
    complexity_score: float       # average complexity
    documentation_coverage: float # 0-100
    test_coverage_estimate: float # 0-100


class AdvancedReporter:
    """
    Advanced reporting engine that generates comprehensive
    ERROR/CODEBASE-STATE reports with detailed analysis
    """
    
    def __init__(self, graph_manager: GraphSitterManager):
        self.graph_manager = graph_manager
        self.error_detector = AdvancedErrorDetector(graph_manager)
        
    def generate_comprehensive_report(self) -> str:
        """
        Generate the advanced ERROR/CODEBASE-STATE report
        that goes far beyond basic summaries
        """
        if not self.graph_manager.codebase:
            return "❌ No codebase available for analysis"
        
        # Get comprehensive analysis
        analysis = self.graph_manager.get_comprehensive_analysis()
        
        # Detect all errors
        all_errors = self.error_detector.detect_comprehensive_errors(self.graph_manager.codebase)
        error_summary = self.error_detector.generate_error_summary(all_errors)
        
        # Calculate health metrics
        health_metrics = self._calculate_health_metrics(analysis, all_errors)
        
        # Generate report sections
        report_sections = [
            self._generate_header(),
            self._generate_executive_summary(analysis, error_summary, health_metrics),
            self._generate_error_enumeration(all_errors),
            self._generate_critical_files_analysis(error_summary, analysis),
            self._generate_entry_points_analysis(analysis),
            self._generate_dependency_health_analysis(analysis),
            self._generate_code_quality_dashboard(health_metrics),
            self._generate_fix_prioritization(all_errors),
            self._generate_recommendations(),
            self._generate_footer()
        ]
        
        return "\n\n".join(report_sections)
    
    def _generate_header(self) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""
# 🚨 ADVANCED ERROR/CODEBASE-STATE ANALYSIS REPORT 🚨
{'=' * 70}

**Repository:** Zeeeepa/codebase-analytics
**Analysis Engine:** Graph-sitter Enhanced with Tree-sitter AST Analysis
**Report Generated:** {timestamp}
**Analysis Type:** Comprehensive Error Detection & Codebase Health Assessment
"""
    
    def _generate_executive_summary(self, analysis: Dict, error_summary: Dict, health_metrics: CodebaseHealthMetrics) -> str:
        """Generate executive summary with key insights"""
        summary = analysis.get('summary', {})
        
        # Calculate key metrics
        total_errors = error_summary.get('total_errors', 0)
        critical_errors = error_summary.get('by_severity', {}).get('critical', 0)
        major_errors = error_summary.get('by_severity', {}).get('major', 0)
        
        # Determine overall health status
        if critical_errors > 0:
            health_status = "🔴 CRITICAL"
        elif major_errors > 5:
            health_status = "🟠 NEEDS ATTENTION"
        elif total_errors > 10:
            health_status = "🟡 MODERATE"
        else:
            health_status = "🟢 HEALTHY"
        
        return f"""
## 📊 EXECUTIVE SUMMARY

### 🎯 **Codebase Health Status: {health_status}**

### 📈 **Key Metrics:**
- **📁 Total Files:** {summary.get('total_files', 0)}
- **🔧 Total Functions:** {summary.get('total_functions', 0)}
- **🎯 Entry Points:** {len(analysis.get('entry_points', []))}
- **🚨 Total Errors:** {total_errors}
- **⚠️ Critical Errors:** {critical_errors}
- **👉 Major Errors:** {major_errors}
- **🔍 Minor Issues:** {error_summary.get('by_severity', {}).get('minor', 0)}

### 🏥 **Health Metrics:**
- **🔧 Maintainability Score:** {health_metrics.maintainability_score:.1f}/100
- **💸 Technical Debt Ratio:** {health_metrics.technical_debt_ratio:.1%}
- **🐛 Error Density:** {health_metrics.error_density:.1f} errors/1000 LOC
- **📚 Documentation Coverage:** {health_metrics.documentation_coverage:.1f}%

### 🎯 **Critical Insights:**
{self._generate_critical_insights(analysis, error_summary, health_metrics)}
"""
    
    def _generate_critical_insights(self, analysis: Dict, error_summary: Dict, health_metrics: CodebaseHealthMetrics) -> str:
        """Generate critical insights from analysis"""
        insights = []
        
        # Error concentration analysis
        critical_files = error_summary.get('critical_files', [])
        if critical_files:
            top_file, error_count = critical_files[0]
            insights.append(f"- **🎯 Error Hotspot:** `{top_file}` contains {error_count} errors ({error_count/error_summary.get('total_errors', 1)*100:.1f}% of total)")
        
        # Most common error types
        common_errors = error_summary.get('most_common_errors', [])
        if common_errors:
            top_error_type, count = common_errors[0]
            insights.append(f"- **🔥 Most Common Issue:** {top_error_type.replace('_', ' ').title()} ({count} occurrences)")
        
        # Entry point analysis
        entry_points = analysis.get('entry_points', [])
        api_endpoints = [ep for ep in entry_points if ep.get('type') == 'api_endpoint']
        if api_endpoints:
            insights.append(f"- **🌐 API Surface:** {len(api_endpoints)} API endpoints detected")
        
        # Complexity insights
        if health_metrics.technical_debt_ratio > 0.3:
            insights.append(f"- **⚠️ High Technical Debt:** {health_metrics.technical_debt_ratio:.1%} of codebase needs refactoring")
        
        return "\n".join(insights) if insights else "- **✅ No critical insights detected**"
    
    def _generate_error_enumeration(self, errors: List[CodeError]) -> str:
        """Generate comprehensive error enumeration - LIST ALL ERRORS 1 by 1"""
        if not errors:
            return "## 🎉 NO ERRORS DETECTED\n\nCongratulations! No errors were found in the codebase."
        
        lines = [
            "## 🚨 COMPREHENSIVE ERROR ENUMERATION",
            f"**Total Errors Found:** {len(errors)}",
            "",
            "### 📋 **All Errors Listed Individually:**"
        ]
        
        # Group errors by severity for organized presentation
        by_severity = {}
        for error in errors:
            severity = error.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(error)
        
        error_num = 1
        
        # Process each severity level
        for severity in [ErrorSeverity.CRITICAL, ErrorSeverity.MAJOR, ErrorSeverity.MINOR, ErrorSeverity.INFO]:
            severity_errors = by_severity.get(severity.value, [])
            if not severity_errors:
                continue
            
            # Severity section header
            severity_emoji = {
                'critical': '🔴',
                'major': '🟠', 
                'minor': '🟡',
                'info': '🔵'
            }
            
            lines.append(f"\n#### {severity_emoji.get(severity.value, '⚪')} **{severity.value.upper()} ERRORS ({len(severity_errors)})**")
            lines.append("")
            
            # List each error with full context
            for error in severity_errors:
                lines.extend([
                    f"**{error_num}. [{error.severity.value.upper()}] {error.message}**",
                    f"   - **📁 File:** `{error.file_path}:{error.line_number}`",
                    f"   - **🏷️ Category:** {error.category.value.title()}",
                    f"   - **🎯 Impact:** {error.impact_assessment}",
                    f"   - **🔧 Fix:** {error.fix_suggestion}",
                    f"   - **🔗 Affects:** {', '.join(error.affected_symbols) if error.affected_symbols else 'N/A'}",
                    f"   - **📊 Confidence:** {error.confidence:.0%}",
                    ""
                ])
                
                # Add context if available
                if error.context:
                    context_str = self._format_context(error.context)
                    if context_str:
                        lines.append(f"   - **📋 Context:** {context_str}")
                        lines.append("")
                
                error_num += 1
        
        return "\n".join(lines)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format error context for display"""
        if not context:
            return ""
        
        formatted_parts = []
        for key, value in context.items():
            if key == "line" and value:
                formatted_parts.append(f"Line: `{value}`")
            elif key == "pattern" and value:
                formatted_parts.append(f"Pattern: `{value}`")
            elif key == "function_name" and value:
                formatted_parts.append(f"Function: `{value}`")
            elif key == "variable" and value:
                formatted_parts.append(f"Variable: `{value}`")
            elif key == "line_count" and value:
                formatted_parts.append(f"Lines: {value}")
        
        return ", ".join(formatted_parts)
    
    def _generate_critical_files_analysis(self, error_summary: Dict, analysis: Dict) -> str:
        """Generate analysis of most critical files"""
        critical_files = error_summary.get('critical_files', [])
        
        if not critical_files:
            return "## 📁 CRITICAL FILES ANALYSIS\n\n✅ **No files with significant error concentrations detected.**"
        
        lines = [
            "## 📁 CRITICAL FILES ANALYSIS",
            "",
            "### 🎯 **Files Requiring Immediate Attention:**"
        ]
        
        for i, (file_path, error_count) in enumerate(critical_files, 1):
            file_data = error_summary.get('by_file', {}).get(file_path, {})
            file_errors = file_data.get('errors', [])
            
            # Analyze error types in this file
            error_types = {}
            critical_count = 0
            major_count = 0
            
            for error in file_errors:
                error_types[error.category.value] = error_types.get(error.category.value, 0) + 1
                if error.severity == ErrorSeverity.CRITICAL:
                    critical_count += 1
                elif error.severity == ErrorSeverity.MAJOR:
                    major_count += 1
            
            lines.extend([
                f"\n**{i}. `{file_path}`**",
                f"   - **🚨 Total Errors:** {error_count}",
                f"   - **⚠️ Critical:** {critical_count} | **👉 Major:** {major_count}",
                f"   - **🏷️ Error Types:** {', '.join(error_types.keys())}",
                f"   - **📊 Error Density:** {error_count} errors in file",
                ""
            ])
        
        return "\n".join(lines)
    
    def _generate_entry_points_analysis(self, analysis: Dict) -> str:
        """Generate analysis of entry points and most important functions"""
        entry_points = analysis.get('entry_points', [])
        important_functions = analysis.get('important_functions', [])
        
        lines = [
            "## 🎯 ENTRY POINTS & CRITICAL FUNCTIONS ANALYSIS",
            ""
        ]
        
        if entry_points:
            lines.extend([
                "### 🚀 **Entry Points Detected:**",
                ""
            ])
            
            for i, ep in enumerate(entry_points, 1):
                lines.extend([
                    f"**{i}. `{ep.get('name', 'unknown')}`**",
                    f"   - **📁 File:** `{ep.get('file_path', 'unknown')}`",
                    f"   - **🏷️ Type:** {ep.get('type', 'unknown').replace('_', ' ').title()}",
                    f"   - **📊 Usage Count:** {ep.get('usage_count', 0)}",
                    f"   - **🔧 Complexity:** {ep.get('complexity', 0):.1f}",
                    ""
                ])
        
        if important_functions:
            lines.extend([
                "### 🌟 **Most Important Functions:**",
                ""
            ])
            
            for i, func in enumerate(important_functions[:10], 1):
                entry_marker = "🎯" if func.is_entry_point else "🔧"
                lines.extend([
                    f"**{i}. {entry_marker} `{func.name}`** (Score: {func.importance_score}/100)",
                    f"   - **📁 File:** `{func.file_path}`",
                    f"   - **📊 Usage:** {func.usage_count} | **🔗 Dependencies:** {func.dependency_count}",
                    f"   - **📞 Calls:** {func.call_count} | **🎯 Entry Point:** {func.is_entry_point}",
                    f"   - **🔧 Complexity:** {func.complexity_score:.1f}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def _generate_dependency_health_analysis(self, analysis: Dict) -> str:
        """Generate dependency health analysis"""
        dependency_health = analysis.get('dependency_health', {})
        
        lines = [
            "## 🔗 DEPENDENCY HEALTH ANALYSIS",
            "",
            f"### 📊 **Dependency Metrics:**",
            f"- **🌐 External Dependencies:** {dependency_health.get('external_dependency_count', 0)}",
            f"- **🔄 Circular Dependencies:** {len(dependency_health.get('circular_dependencies', []))}",
            f"- **❌ Missing Imports:** {len(dependency_health.get('missing_imports', []))}",
            f"- **🗑️ Unused Imports:** {len(dependency_health.get('unused_imports', []))}",
            f"- **📏 Dependency Depth:** {dependency_health.get('dependency_depth', 0)}",
            ""
        ]
        
        # Add specific issues if found
        circular_deps = dependency_health.get('circular_dependencies', [])
        if circular_deps:
            lines.extend([
                "### ⚠️ **Circular Dependencies Detected:**",
                ""
            ])
            for i, dep in enumerate(circular_deps, 1):
                lines.append(f"{i}. {dep}")
            lines.append("")
        
        missing_imports = dependency_health.get('missing_imports', [])
        if missing_imports:
            lines.extend([
                "### ❌ **Missing Imports:**",
                ""
            ])
            for i, imp in enumerate(missing_imports, 1):
                lines.append(f"{i}. {imp}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_code_quality_dashboard(self, health_metrics: CodebaseHealthMetrics) -> str:
        """Generate code quality dashboard"""
        return f"""
## 📊 CODE QUALITY DASHBOARD

### 🏥 **Health Metrics Overview:**

| Metric | Score | Status |
|--------|-------|--------|
| 🔧 Maintainability | {health_metrics.maintainability_score:.1f}/100 | {self._get_status_emoji(health_metrics.maintainability_score)} |
| 💸 Technical Debt | {health_metrics.technical_debt_ratio:.1%} | {self._get_debt_status_emoji(health_metrics.technical_debt_ratio)} |
| 🐛 Error Density | {health_metrics.error_density:.1f}/1000 LOC | {self._get_error_density_status(health_metrics.error_density)} |
| 📚 Documentation | {health_metrics.documentation_coverage:.1f}% | {self._get_status_emoji(health_metrics.documentation_coverage)} |
| 🧪 Test Coverage (Est.) | {health_metrics.test_coverage_estimate:.1f}% | {self._get_status_emoji(health_metrics.test_coverage_estimate)} |

### 📈 **Quality Trends:**
- **Overall Code Health:** {self._calculate_overall_health(health_metrics)}/100
- **Improvement Priority:** {self._get_improvement_priority(health_metrics)}
"""
    
    def _generate_fix_prioritization(self, errors: List[CodeError]) -> str:
        """Generate fix prioritization based on impact and effort"""
        if not errors:
            return "## 🎯 FIX PRIORITIZATION\n\n✅ **No fixes required - codebase is clean!**"
        
        # Sort errors by priority (severity + impact)
        prioritized_errors = sorted(errors, key=lambda e: (
            {'critical': 4, 'major': 3, 'minor': 2, 'info': 1}[e.severity.value],
            e.confidence
        ), reverse=True)
        
        lines = [
            "## 🎯 FIX PRIORITIZATION",
            "",
            "### 🚨 **Immediate Action Required (Critical & High-Impact):**"
        ]
        
        # High priority fixes
        high_priority = [e for e in prioritized_errors if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.MAJOR]][:5]
        
        for i, error in enumerate(high_priority, 1):
            effort_estimate = self._estimate_fix_effort(error)
            lines.extend([
                f"\n**{i}. {error.message}**",
                f"   - **📁 Location:** `{error.file_path}:{error.line_number}`",
                f"   - **⏱️ Estimated Effort:** {effort_estimate}",
                f"   - **🎯 Impact:** {error.impact_assessment}",
                f"   - **🔧 Fix:** {error.fix_suggestion}",
                ""
            ])
        
        # Medium priority fixes
        medium_priority = [e for e in prioritized_errors if e.severity == ErrorSeverity.MINOR][:3]
        if medium_priority:
            lines.extend([
                "### 🟡 **Medium Priority (When Time Permits):**",
                ""
            ])
            
            for i, error in enumerate(medium_priority, 1):
                lines.extend([
                    f"**{i}. {error.message}** - `{error.file_path}:{error.line_number}`",
                    f"   - **Fix:** {error.fix_suggestion}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations"""
        return """
## 💡 ACTIONABLE RECOMMENDATIONS

### 🎯 **Immediate Actions:**
1. **🔴 Address Critical Errors:** Fix all critical errors immediately to prevent runtime failures
2. **🟠 Resolve Major Issues:** Tackle major errors that affect functionality and reliability
3. **📁 Focus on Error Hotspots:** Prioritize files with highest error concentrations

### 🔧 **Code Quality Improvements:**
1. **📚 Add Documentation:** Improve documentation coverage for better maintainability
2. **🧪 Increase Test Coverage:** Add tests for critical functions and entry points
3. **🔄 Refactor Long Functions:** Break down complex functions into smaller, manageable pieces

### 📊 **Monitoring & Prevention:**
1. **🤖 Automated Checks:** Implement pre-commit hooks to catch errors early
2. **📈 Regular Analysis:** Run this analysis regularly to track code quality trends
3. **👥 Code Reviews:** Establish code review processes to prevent error introduction

### 🚀 **Performance Optimizations:**
1. **⚡ Address Performance Issues:** Fix inefficient patterns identified in the analysis
2. **🔗 Optimize Dependencies:** Review and optimize dependency usage
3. **🧹 Remove Dead Code:** Clean up unused code and imports
"""
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """
---

## 📋 **Report Information**

**Analysis Engine:** Graph-sitter Enhanced with Tree-sitter AST Analysis  
**Error Detection:** Comprehensive multi-layer analysis (Syntax, Semantic, Structural, Implementation)  
**Confidence Levels:** Based on pattern matching accuracy and context analysis  
**Recommendations:** Prioritized by impact, effort, and business value  

**🔄 Next Analysis:** Re-run after implementing fixes to track improvements  
**📞 Support:** For questions about this analysis, refer to the Graph-sitter documentation  

---

*This report was generated automatically by the Enhanced Graph-sitter Analysis Engine*
"""
    
    def _calculate_health_metrics(self, analysis: Dict, errors: List[CodeError]) -> CodebaseHealthMetrics:
        """Calculate comprehensive health metrics"""
        summary = analysis.get('summary', {})
        total_functions = summary.get('total_functions', 1)
        total_files = summary.get('total_files', 1)
        
        # Calculate error density (errors per 1000 lines of code)
        # Estimate LOC based on functions (rough estimate)
        estimated_loc = total_functions * 20  # Assume 20 lines per function average
        error_density = (len(errors) / max(estimated_loc, 1)) * 1000
        
        # Calculate technical debt ratio
        critical_errors = len([e for e in errors if e.severity == ErrorSeverity.CRITICAL])
        major_errors = len([e for e in errors if e.severity == ErrorSeverity.MAJOR])
        technical_debt_ratio = min((critical_errors * 0.1 + major_errors * 0.05), 1.0)
        
        # Calculate maintainability score
        maintainability_score = max(100 - (critical_errors * 10 + major_errors * 5 + len(errors) * 1), 0)
        
        # Estimate documentation coverage
        doc_errors = len([e for e in errors if e.category == ErrorCategory.STYLE and 'documentation' in e.message.lower()])
        documentation_coverage = max(100 - (doc_errors / max(total_functions, 1)) * 100, 0)
        
        # Estimate test coverage (very rough estimate)
        test_coverage_estimate = max(60 - error_density, 0)  # Rough correlation
        
        return CodebaseHealthMetrics(
            maintainability_score=maintainability_score,
            technical_debt_ratio=technical_debt_ratio,
            error_density=error_density,
            complexity_score=1.0,  # Placeholder
            documentation_coverage=documentation_coverage,
            test_coverage_estimate=test_coverage_estimate
        )
    
    def _get_status_emoji(self, score: float) -> str:
        """Get status emoji based on score"""
        if score >= 80:
            return "🟢 Excellent"
        elif score >= 60:
            return "🟡 Good"
        elif score >= 40:
            return "🟠 Needs Improvement"
        else:
            return "🔴 Poor"
    
    def _get_debt_status_emoji(self, ratio: float) -> str:
        """Get technical debt status emoji"""
        if ratio <= 0.1:
            return "🟢 Low"
        elif ratio <= 0.3:
            return "🟡 Moderate"
        elif ratio <= 0.5:
            return "🟠 High"
        else:
            return "🔴 Critical"
    
    def _get_error_density_status(self, density: float) -> str:
        """Get error density status"""
        if density <= 5:
            return "🟢 Low"
        elif density <= 15:
            return "🟡 Moderate"
        elif density <= 30:
            return "🟠 High"
        else:
            return "🔴 Critical"
    
    def _calculate_overall_health(self, metrics: CodebaseHealthMetrics) -> int:
        """Calculate overall health score"""
        weights = {
            'maintainability': 0.3,
            'technical_debt': 0.3,
            'error_density': 0.2,
            'documentation': 0.2
        }
        
        # Invert technical debt and error density (lower is better)
        debt_score = max(100 - metrics.technical_debt_ratio * 100, 0)
        density_score = max(100 - metrics.error_density * 2, 0)
        
        overall = (
            metrics.maintainability_score * weights['maintainability'] +
            debt_score * weights['technical_debt'] +
            density_score * weights['error_density'] +
            metrics.documentation_coverage * weights['documentation']
        )
        
        return int(overall)
    
    def _get_improvement_priority(self, metrics: CodebaseHealthMetrics) -> str:
        """Get improvement priority recommendation"""
        if metrics.technical_debt_ratio > 0.4:
            return "🔴 Technical Debt Reduction"
        elif metrics.error_density > 20:
            return "🟠 Error Resolution"
        elif metrics.documentation_coverage < 50:
            return "📚 Documentation"
        elif metrics.maintainability_score < 60:
            return "🔧 Code Refactoring"
        else:
            return "🟢 Maintenance & Monitoring"
    
    def _estimate_fix_effort(self, error: CodeError) -> str:
        """Estimate effort required to fix an error"""
        if error.severity == ErrorSeverity.CRITICAL:
            return "🔴 High (2-4 hours)"
        elif error.severity == ErrorSeverity.MAJOR:
            return "🟠 Medium (1-2 hours)"
        elif error.severity == ErrorSeverity.MINOR:
            return "🟡 Low (15-30 minutes)"
        else:
            return "🟢 Minimal (5-15 minutes)"
