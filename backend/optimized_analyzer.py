"""
Optimized Graph-sitter Analyzer
Main entry point for the enhanced backend analysis system
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the backend directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graph_manager import GraphSitterManager
from detectors.error_engine import AdvancedErrorDetector
from reporting.advanced_reporter import AdvancedReporter


class OptimizedAnalyzer:
    """
    Optimized analyzer that leverages Graph-sitter's full capabilities
    for comprehensive codebase analysis and error detection
    """
    
    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.graph_manager = GraphSitterManager(str(self.codebase_path))
        self.error_detector = None
        self.reporter = None
        
    def initialize(self) -> bool:
        """Initialize the analyzer components"""
        if not self.graph_manager.initialize():
            return False
        
        self.error_detector = AdvancedErrorDetector(self.graph_manager)
        self.reporter = AdvancedReporter(self.graph_manager)
        return True
    
    def analyze_comprehensive(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all enhanced capabilities
        """
        if not self.graph_manager.codebase:
            return {"error": "Codebase not initialized"}
        
        # Get comprehensive analysis from graph manager
        analysis = self.graph_manager.get_comprehensive_analysis()
        
        # Detect all errors using advanced error detector
        all_errors = self.error_detector.detect_comprehensive_errors(self.graph_manager.codebase)
        error_summary = self.error_detector.generate_error_summary(all_errors)
        
        # Combine results
        comprehensive_results = {
            **analysis,
            "all_errors": all_errors,
            "error_summary": error_summary,
            "error_enumeration_report": self.graph_manager.get_error_enumeration_report()
        }
        
        return comprehensive_results
    
    def generate_advanced_report(self) -> str:
        """
        Generate the advanced ERROR/CODEBASE-STATE report
        """
        if not self.reporter:
            return "❌ Reporter not initialized"
        
        return self.reporter.generate_comprehensive_report()
    
    def list_all_errors_individually(self) -> List[Dict[str, Any]]:
        """
        List ALL errors one by one with full context
        """
        if not self.error_detector:
            return []
        
        all_errors = self.error_detector.detect_comprehensive_errors(self.graph_manager.codebase)
        
        # Convert to detailed dictionaries
        error_list = []
        for i, error in enumerate(all_errors, 1):
            error_dict = {
                "error_number": i,
                "id": error.id,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "file_path": error.file_path,
                "line_number": error.line_number,
                "column_number": error.column_number,
                "context": error.context,
                "affected_symbols": error.affected_symbols,
                "dependencies": error.dependencies,
                "fix_suggestion": error.fix_suggestion,
                "impact_assessment": error.impact_assessment,
                "confidence": error.confidence
            }
            error_list.append(error_dict)
        
        return error_list
    
    def get_most_important_functions_and_entry_points(self) -> Dict[str, Any]:
        """
        Get most important code files and entry points
        """
        if not self.graph_manager.codebase:
            return {}
        
        analysis = self.graph_manager.get_comprehensive_analysis()
        
        return {
            "most_important_functions": analysis.get("important_functions", []),
            "entry_points": analysis.get("entry_points", []),
            "summary": analysis.get("summary", {}),
            "dependency_health": analysis.get("dependency_health", {})
        }
    
    def run_complete_analysis(self) -> str:
        """
        Run complete analysis and return formatted report
        """
        print("🚀 Starting Enhanced Graph-sitter Analysis...")
        
        if not self.initialize():
            return "❌ Failed to initialize analyzer"
        
        print("✅ Analyzer initialized successfully")
        print("🔍 Performing comprehensive analysis...")
        
        # Generate the advanced report
        report = self.generate_advanced_report()
        
        print("📊 Analysis complete!")
        return report


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Graph-sitter Codebase Analyzer")
    parser.add_argument("path", help="Path to codebase to analyze")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--errors-only", action="store_true", help="Show only error enumeration")
    parser.add_argument("--functions-only", action="store_true", help="Show only important functions")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OptimizedAnalyzer(args.path)
    
    if args.errors_only:
        # List all errors individually
        if not analyzer.initialize():
            print("❌ Failed to initialize analyzer")
            return
        
        errors = analyzer.list_all_errors_individually()
        print(f"\n🚨 COMPREHENSIVE ERROR ENUMERATION ({len(errors)} errors):")
        print("=" * 60)
        
        for error in errors:
            print(f"\n{error['error_number']}. [{error['severity'].upper()}] {error['message']}")
            print(f"   📁 File: {error['file_path']}:{error['line_number']}")
            print(f"   🏷️ Category: {error['category']}")
            print(f"   🎯 Impact: {error['impact_assessment']}")
            print(f"   🔧 Fix: {error['fix_suggestion']}")
            print(f"   📊 Confidence: {error['confidence']:.0%}")
            
            if error['context']:
                print(f"   📋 Context: {error['context']}")
    
    elif args.functions_only:
        # Show important functions and entry points
        if not analyzer.initialize():
            print("❌ Failed to initialize analyzer")
            return
        
        functions_data = analyzer.get_most_important_functions_and_entry_points()
        
        print("\n🌟 MOST IMPORTANT FUNCTIONS:")
        print("=" * 40)
        
        for i, func in enumerate(functions_data.get("most_important_functions", [])[:10], 1):
            entry_marker = "🎯" if func.is_entry_point else "🔧"
            print(f"{i}. {entry_marker} {func.name} (Score: {func.importance_score}/100)")
            print(f"   📁 File: {func.file_path}")
            print(f"   📊 Usage: {func.usage_count} | Dependencies: {func.dependency_count}")
            print()
        
        print("\n🎯 ENTRY POINTS:")
        print("=" * 20)
        
        for i, ep in enumerate(functions_data.get("entry_points", []), 1):
            print(f"{i}. {ep.get('name', 'unknown')} ({ep.get('type', 'unknown')})")
            print(f"   📁 File: {ep.get('file_path', 'unknown')}")
            print(f"   📊 Usage: {ep.get('usage_count', 0)}")
            print()
    
    else:
        # Full comprehensive report
        report = analyzer.run_complete_analysis()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()
