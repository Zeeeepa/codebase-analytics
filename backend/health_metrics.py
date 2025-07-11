"""
Codebase Health Metrics Calculator
Comprehensive health assessment and trend analysis
"""

from typing import Dict, List, Any
from collections import defaultdict
import math


class CodebaseHealthCalculator:
    """Calculator for codebase health metrics and trends"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        
    def calculate_overall_health(self, issues: List[Any], complexity_metrics: Dict[str, Any], 
                               dead_code_analysis: Dict[str, Any]) -> float:
        """Calculate overall health score (0-100)"""
        print("🏥 Calculating health metrics...")
        
        # Base score
        health_score = 100.0
        
        # Issue penalties
        issue_penalty = self._calculate_issue_penalty(issues)
        health_score -= issue_penalty
        
        # Complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(complexity_metrics)
        health_score -= complexity_penalty
        
        # Dead code penalty
        dead_code_penalty = self._calculate_dead_code_penalty(dead_code_analysis)
        health_score -= dead_code_penalty
        
        # Maintainability bonus/penalty
        maintainability_adjustment = self._calculate_maintainability_adjustment()
        health_score += maintainability_adjustment
        
        return max(0, min(100, round(health_score, 1)))
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze health trends (simplified - would need historical data)"""
        # This would typically analyze historical data
        # For now, we'll provide current state analysis
        
        return {
            'current_state': 'stable',
            'trend_direction': 'neutral',
            'key_indicators': {
                'code_growth_rate': 'moderate',
                'issue_introduction_rate': 'low',
                'complexity_trend': 'stable',
                'maintainability_trend': 'improving'
            },
            'recommendations': [
                'Continue current practices',
                'Monitor complexity growth',
                'Address critical issues promptly'
            ]
        }
    
    def assess_risks(self, issues: List[Any], circular_deps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess various risks in the codebase"""
        risks = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'critical_risks': [],
            'mitigation_strategies': []
        }
        
        # Critical issue risks
        critical_issues = [i for i in issues if i.severity.value == 'critical']
        if len(critical_issues) > 5:
            risks['risk_factors'].append('High number of critical issues')
            risks['critical_risks'].append({
                'type': 'critical_issues',
                'severity': 'high',
                'count': len(critical_issues),
                'description': f'{len(critical_issues)} critical issues need immediate attention'
            })
        
        # Circular dependency risks
        if len(circular_deps) > 0:
            risks['risk_factors'].append('Circular dependencies detected')
            risks['critical_risks'].append({
                'type': 'circular_dependencies',
                'severity': 'medium',
                'count': len(circular_deps),
                'description': f'{len(circular_deps)} circular dependencies may complicate maintenance'
            })
        
        # Calculate overall risk level
        if len(risks['critical_risks']) > 2:
            risks['overall_risk_level'] = 'high'
        elif len(risks['critical_risks']) > 0:
            risks['overall_risk_level'] = 'medium'
        
        # Generate mitigation strategies
        risks['mitigation_strategies'] = self._generate_mitigation_strategies(risks['critical_risks'])
        
        return risks
    
    def _calculate_issue_penalty(self, issues: List[Any]) -> float:
        """Calculate penalty based on issues"""
        if not issues:
            return 0.0
        
        penalty = 0.0
        severity_weights = {
            'critical': 10.0,
            'major': 5.0,
            'minor': 2.0,
            'info': 0.5
        }
        
        for issue in issues:
            penalty += severity_weights.get(issue.severity.value, 1.0)
        
        # Normalize by codebase size
        total_functions = len(self.codebase.functions)
        if total_functions > 0:
            penalty = penalty / total_functions * 10
        
        return min(penalty, 50.0)  # Cap at 50 points
    
    def _calculate_complexity_penalty(self, complexity_metrics: Dict[str, Any]) -> float:
        """Calculate penalty based on complexity"""
        if not complexity_metrics:
            return 0.0
        
        penalty = 0.0
        
        # Average complexity penalty
        avg_complexity = complexity_metrics.get('average_complexity', 0)
        if avg_complexity > 10:
            penalty += (avg_complexity - 10) * 2
        
        # High complexity functions penalty
        high_complexity_count = complexity_metrics.get('high_complexity_functions', 0)
        penalty += high_complexity_count * 3
        
        return min(penalty, 30.0)  # Cap at 30 points
    
    def _calculate_dead_code_penalty(self, dead_code_analysis: Dict[str, Any]) -> float:
        """Calculate penalty based on dead code"""
        if not dead_code_analysis:
            return 0.0
        
        total_dead = dead_code_analysis.get('summary', {}).get('total_dead_code_items', 0)
        total_functions = len(self.codebase.functions)
        
        if total_functions == 0:
            return 0.0
        
        dead_percentage = (total_dead / total_functions) * 100
        penalty = dead_percentage * 0.2  # 0.2 points per percent of dead code
        
        return min(penalty, 15.0)  # Cap at 15 points
    
    def _calculate_maintainability_adjustment(self) -> float:
        """Calculate maintainability adjustment (bonus/penalty)"""
        # This would analyze code structure, documentation, etc.
        # For now, we'll provide a neutral adjustment
        return 0.0
    
    def _generate_mitigation_strategies(self, critical_risks: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation strategies for identified risks"""
        strategies = []
        
        for risk in critical_risks:
            if risk['type'] == 'critical_issues':
                strategies.append('Prioritize fixing critical issues in next sprint')
                strategies.append('Implement automated testing to prevent critical issues')
            elif risk['type'] == 'circular_dependencies':
                strategies.append('Refactor circular dependencies using dependency injection')
                strategies.append('Implement architectural guidelines to prevent future cycles')
        
        # General strategies
        if len(critical_risks) > 0:
            strategies.extend([
                'Establish code review process',
                'Implement continuous integration with quality gates',
                'Regular technical debt assessment'
            ])
        
        return strategies
    
    def calculate_technical_debt_hours(self, issues: List[Any]) -> Dict[str, Any]:
        """Estimate technical debt in hours"""
        debt_hours = 0.0
        
        # Effort estimates by issue type and severity
        effort_matrix = {
            'critical': {'null_reference': 4, 'type_mismatch': 6, 'undefined_variable': 2},
            'major': {'long_function': 8, 'code_duplication': 6, 'improper_exception_handling': 4},
            'minor': {'missing_documentation': 1, 'magic_number': 0.5, 'inconsistent_naming': 2},
            'info': {'line_length_violation': 0.25, 'import_organization': 0.5}
        }
        
        for issue in issues:
            severity = issue.severity.value
            issue_type = issue.issue_type.value
            
            # Get effort estimate
            if severity in effort_matrix and issue_type in effort_matrix[severity]:
                debt_hours += effort_matrix[severity][issue_type]
            else:
                # Default estimates by severity
                default_efforts = {'critical': 4, 'major': 2, 'minor': 1, 'info': 0.5}
                debt_hours += default_efforts.get(severity, 1)
        
        return {
            'total_hours': round(debt_hours, 1),
            'total_days': round(debt_hours / 8, 1),
            'estimated_cost': round(debt_hours * 100, 0),  # Assuming $100/hour
            'breakdown_by_severity': self._calculate_debt_breakdown(issues, effort_matrix)
        }
    
    def _calculate_debt_breakdown(self, issues: List[Any], effort_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate technical debt breakdown by severity"""
        breakdown = defaultdict(float)
        
        for issue in issues:
            severity = issue.severity.value
            issue_type = issue.issue_type.value
            
            if severity in effort_matrix and issue_type in effort_matrix[severity]:
                breakdown[severity] += effort_matrix[severity][issue_type]
            else:
                default_efforts = {'critical': 4, 'major': 2, 'minor': 1, 'info': 0.5}
                breakdown[severity] += default_efforts.get(severity, 1)
        
        return dict(breakdown)
    
    def generate_health_report(self, overall_health: float, issues: List[Any], 
                             complexity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        # Calculate grade
        if overall_health >= 90:
            grade = 'A'
            status = 'Excellent'
        elif overall_health >= 80:
            grade = 'B'
            status = 'Good'
        elif overall_health >= 70:
            grade = 'C'
            status = 'Fair'
        elif overall_health >= 60:
            grade = 'D'
            status = 'Poor'
        else:
            grade = 'F'
            status = 'Critical'
        
        # Key metrics
        critical_issues = len([i for i in issues if i.severity.value == 'critical'])
        avg_complexity = complexity_metrics.get('average_complexity', 0)
        
        # Recommendations
        recommendations = []
        if critical_issues > 0:
            recommendations.append(f'Address {critical_issues} critical issues immediately')
        if avg_complexity > 15:
            recommendations.append('Refactor high-complexity functions')
        if overall_health < 70:
            recommendations.append('Implement comprehensive code review process')
        
        return {
            'overall_score': overall_health,
            'grade': grade,
            'status': status,
            'key_metrics': {
                'total_issues': len(issues),
                'critical_issues': critical_issues,
                'average_complexity': round(avg_complexity, 1),
                'total_functions': len(self.codebase.functions),
                'total_files': len(self.codebase.files)
            },
            'recommendations': recommendations,
            'next_actions': self._generate_next_actions(overall_health, critical_issues),
            'technical_debt': self.calculate_technical_debt_hours(issues)
        }
    
    def _generate_next_actions(self, health_score: float, critical_issues: int) -> List[str]:
        """Generate next actions based on health score"""
        actions = []
        
        if critical_issues > 0:
            actions.append('Fix critical issues within 24 hours')
        
        if health_score < 50:
            actions.append('Conduct emergency code review')
            actions.append('Implement immediate quality gates')
        elif health_score < 70:
            actions.append('Schedule technical debt sprint')
            actions.append('Increase code review coverage')
        else:
            actions.append('Maintain current quality practices')
            actions.append('Monitor for new issues')
        
        return actions

