# Codebase Consolidation Report

Generated: unified_api.py

## Source Files
- backend/comprehensive_analysis.py
- backend/api.py
- backend/analyzer.py

## Consolidated Elements

### Classes
- `IssueType` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `IssueSeverity` from backend/analyzer.py - ✅ INCLUDED
- `Issue` from backend/analyzer.py - ✅ INCLUDED
- `ComprehensiveAnalyzer` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `CodebaseStats` from backend/api.py - ✅ INCLUDED
- `FileTestStats` from backend/api.py - ✅ INCLUDED
- `FunctionContext` from backend/api.py - ✅ INCLUDED
- `TestAnalysis` from backend/api.py - ✅ INCLUDED
- `FunctionAnalysis` from backend/api.py - ✅ INCLUDED
- `ClassAnalysis` from backend/api.py - ✅ INCLUDED
- `FileIssue` from backend/api.py - ✅ INCLUDED
- `ExtendedAnalysis` from backend/api.py - ✅ INCLUDED
- `RepoRequest` from backend/api.py - ✅ INCLUDED
- `Symbol` from backend/api.py - ✅ INCLUDED
- `FileNode` from backend/api.py - ✅ INCLUDED
- `AnalysisResponse` from backend/api.py - ✅ INCLUDED
- `AnalysisType` from backend/analyzer.py - ✅ INCLUDED
- `IssueCategory` from backend/analyzer.py - ✅ INCLUDED
- `IssueStatus` from backend/analyzer.py - ✅ INCLUDED
- `ChangeType` from backend/analyzer.py - ✅ INCLUDED
- `TransactionPriority` from backend/analyzer.py - ✅ INCLUDED
- `CodeLocation` from backend/analyzer.py - ✅ INCLUDED
- `IssueCollection` from backend/analyzer.py - ✅ INCLUDED
- `AnalysisSummary` from backend/analyzer.py - ✅ INCLUDED
- `CodeQualityResult` from backend/analyzer.py - ✅ INCLUDED
- `DependencyResult` from backend/analyzer.py - ✅ INCLUDED
- `AnalysisResult` from backend/analyzer.py - ✅ INCLUDED
- `CodebaseAnalyzer` from backend/analyzer.py - ✅ INCLUDED

### Functions
- `main` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `__init__` from backend/analyzer.py - ✅ INCLUDED
- `_get_location` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `__str__` from backend/analyzer.py - ✅ INCLUDED
- `analyze` from backend/analyzer.py - ✅ INCLUDED
- `_analyze_dead_code` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_analyze_parameter_issues` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_analyze_type_annotations` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_analyze_circular_dependencies` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_check_circular_deps` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_analyze_implementation_issues` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_generate_report` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_print_report` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_save_report` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `_save_detailed_summaries` from backend/comprehensive_analysis.py - ✅ INCLUDED
- `get_monthly_commits` from backend/api.py - ✅ INCLUDED
- `calculate_cyclomatic_complexity` from backend/analyzer.py - ✅ INCLUDED
- `cc_rank` from backend/api.py - ✅ INCLUDED
- `calculate_doi` from backend/api.py - ✅ INCLUDED
- `get_operators_and_operands` from backend/api.py - ✅ INCLUDED
- `calculate_halstead_volume` from backend/api.py - ✅ INCLUDED
- `count_lines` from backend/api.py - ✅ INCLUDED
- `calculate_maintainability_index` from backend/api.py - ✅ INCLUDED
- `get_maintainability_rank` from backend/api.py - ✅ INCLUDED
- `get_github_repo_description` from backend/api.py - ✅ INCLUDED
- `find_dead_code` from backend/api.py - ✅ INCLUDED
- `get_max_call_chain` from backend/api.py - ✅ INCLUDED
- `analyze_file_issues` from backend/api.py - ✅ INCLUDED
- `build_repo_structure` from backend/api.py - ✅ INCLUDED
- `get_file_type` from backend/api.py - ✅ INCLUDED
- `get_detailed_symbol_context` from backend/api.py - ✅ INCLUDED
- `get_function_by_id` from backend/api.py - ✅ INCLUDED
- `hop_through_imports` from backend/api.py - ✅ INCLUDED
- `fastapi_modal_app` from backend/api.py - ✅ INCLUDED
- `analyze_statement` from backend/api.py - ✅ INCLUDED
- `analyze_block` from backend/api.py - ✅ INCLUDED
- `build_graph` from backend/api.py - ✅ INCLUDED
- `find_available_port` from backend/api.py - ✅ INCLUDED
- `create_issue` from backend/analyzer.py - ✅ INCLUDED
- `get_codebase_summary` from backend/analyzer.py - ✅ INCLUDED
- `get_dependency_graph` from backend/analyzer.py - ✅ INCLUDED
- `get_symbol_references` from backend/analyzer.py - ✅ INCLUDED
- `analyze_codebase` from backend/analyzer.py - ✅ INCLUDED
- `to_dict` from backend/analyzer.py - ✅ INCLUDED
- `from_dict` from backend/analyzer.py - ✅ INCLUDED
- `__post_init__` from backend/analyzer.py - ✅ INCLUDED
- `file` from backend/analyzer.py - ✅ INCLUDED
- `line` from backend/analyzer.py - ✅ INCLUDED
- `add_issue` from backend/analyzer.py - ✅ INCLUDED
- `add_issues` from backend/analyzer.py - ✅ INCLUDED
- `add_filter` from backend/analyzer.py - ✅ INCLUDED
- `get_issues` from backend/analyzer.py - ✅ INCLUDED
- `group_by_severity` from backend/analyzer.py - ✅ INCLUDED
- `group_by_category` from backend/analyzer.py - ✅ INCLUDED
- `group_by_file` from backend/analyzer.py - ✅ INCLUDED
- `statistics` from backend/analyzer.py - ✅ INCLUDED
- `save_to_file` from backend/analyzer.py - ✅ INCLUDED
- `load_from_file` from backend/analyzer.py - ✅ INCLUDED
- `_init_from_url` from backend/analyzer.py - ✅ INCLUDED
- `_init_from_path` from backend/analyzer.py - ✅ INCLUDED
- `_analyze_code_quality` from backend/analyzer.py - ✅ INCLUDED
- `_analyze_dependencies` from backend/analyzer.py - ✅ INCLUDED
- `_analyze_performance` from backend/analyzer.py - ✅ INCLUDED
- `_find_dead_code` from backend/analyzer.py - ✅ INCLUDED
- `_check_function_parameters` from backend/analyzer.py - ✅ INCLUDED
- `_check_implementations` from backend/analyzer.py - ✅ INCLUDED
- `_analyze_import_dependencies` from backend/analyzer.py - ✅ INCLUDED
- `_find_circular_dependencies` from backend/analyzer.py - ✅ INCLUDED
- `save_results` from backend/analyzer.py - ✅ INCLUDED
- `_generate_html_report` from backend/analyzer.py - ✅ INCLUDED
- `_print_console_report` from backend/analyzer.py - ✅ INCLUDED
- `detect_cycles` from backend/analyzer.py - ✅ INCLUDED

### Import Summary
Total unique imports: 75

## Recommendations
1. Review duplicate functions/classes for potential feature merging
2. Test the unified API thoroughly
3. Update any external references to the original files
4. Consider refactoring for better organization
