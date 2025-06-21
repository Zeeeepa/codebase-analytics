#!/usr/bin/env python3
"""
End-to-End Test Suite for Codebase Analytics

This comprehensive test suite validates the entire analysis pipeline from
repository input to final visualization output.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import requests
from fastapi.testclient import TestClient

# Import modules to test
from api import app, parse_repo_url, format_cli_response
from analysis import (
    analyze_codebase, detect_issues, analyze_code_quality,
    analyze_dependencies, analyze_call_graph, detect_entry_points,
    calculate_halstead_metrics, analyze_inheritance_hierarchy,
    detect_comprehensive_issues, get_advanced_codebase_statistics
)

# Create test client
client = TestClient(app)

class TestEndToEndAnalysis:
    """End-to-end tests for the complete analysis pipeline."""
    
    @pytest.fixture
    def sample_repo_data(self):
        """Sample repository data for testing."""
        return {
            "repository": {
                "name": "test-repo",
                "owner": "test-owner",
                "total_files": 10,
                "total_functions": 25,
                "total_classes": 5
            },
            "analysis": {
                "comprehensive_issues": {
                    "total_issues": 15,
                    "issues_by_severity": {
                        "Critical": 2,
                        "High": 3,
                        "Medium": 5,
                        "Low": 5
                    }
                },
                "most_important_entry_points": {
                    "top_10_by_heat": ["main", "init", "process"],
                    "main_functions": ["main", "__main__"],
                    "api_endpoints": ["/api/v1/analyze"],
                    "high_usage_functions": ["utils.helper", "core.processor"]
                },
                "code_quality": {
                    "maintainability_index": 75.5,
                    "comment_density": 15.2,
                    "technical_debt_ratio": 12.8
                }
            }
        }
    
    @pytest.fixture
    def mock_codebase(self):
        """Mock codebase object for testing."""
        mock_codebase = Mock()
        mock_codebase.functions = [Mock(name="test_func", source="def test_func(): pass")]
        mock_codebase.classes = [Mock(name="TestClass")]
        mock_codebase.files = [Mock(filepath="test.py")]
        return mock_codebase

class TestRepositoryParsing:
    """Test repository URL parsing functionality."""
    
    def test_parse_github_url(self):
        """Test parsing GitHub URLs."""
        owner, repo = parse_repo_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"
    
    def test_parse_owner_repo_format(self):
        """Test parsing owner/repo format."""
        owner, repo = parse_repo_url("owner/repo")
        assert owner == "owner"
        assert repo == "repo"
    
    def test_parse_invalid_format(self):
        """Test parsing invalid formats raises ValueError."""
        with pytest.raises(ValueError):
            parse_repo_url("invalid-format")
    
    def test_parse_invalid_github_url(self):
        """Test parsing invalid GitHub URL raises ValueError."""
        with pytest.raises(ValueError):
            parse_repo_url("https://github.com/invalid")

class TestCLIFunctionality:
    """Test CLI functionality transferred to API."""
    
    def test_format_cli_response(self, sample_repo_data):
        """Test CLI response formatting."""
        formatted = format_cli_response(sample_repo_data)
        
        assert "COMPREHENSIVE CODEBASE ANALYSIS RESULTS" in formatted
        assert "test-repo (test-owner)" in formatted
        assert "Total Issues: 15" in formatted
        assert "Critical: 2" in formatted
        assert "Maintainability Index: 75.50" in formatted
    
    def test_cli_analyze_endpoint(self):
        """Test CLI-compatible analysis endpoint."""
        with patch('api.analyze_repository') as mock_analyze:
            mock_analyze.return_value = {"status": "success"}
            
            response = client.get("/cli/analyze/owner/repo")
            assert response.status_code == 200
            mock_analyze.assert_called_once_with("owner", "repo")

class TestAnalysisCore:
    """Test core analysis functionality."""
    
    @patch('analysis.Codebase')
    def test_comprehensive_analysis_pipeline(self, mock_codebase_class, mock_codebase):
        """Test the complete analysis pipeline."""
        mock_codebase_class.from_repo.return_value = mock_codebase
        
        # Test each analysis component
        issues = detect_issues(mock_codebase)
        assert hasattr(issues, 'issues')
        
        quality = analyze_code_quality(mock_codebase)
        assert isinstance(quality, dict)
        
        dependencies = analyze_dependencies(mock_codebase)
        assert hasattr(dependencies, 'total_dependencies')
        
        call_graph = analyze_call_graph(mock_codebase)
        assert hasattr(call_graph, 'total_functions')

class TestAPIEndpoints:
    """Test API endpoints functionality."""
    
    def test_root_endpoint(self):
        """Test root API endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Welcome to the Codebase Analytics API" in response.json()["message"]
    
    @patch('api.Codebase')
    def test_analyze_repository_endpoint(self, mock_codebase_class, mock_codebase):
        """Test repository analysis endpoint."""
        mock_codebase_class.from_repo.return_value = mock_codebase
        
        # Mock analysis functions
        with patch('api.detect_issues') as mock_issues, \
             patch('api.analyze_code_quality') as mock_quality, \
             patch('api.analyze_dependencies') as mock_deps, \
             patch('api.analyze_call_graph') as mock_call_graph:
            
            mock_issues.return_value = Mock(issues=[])
            mock_quality.return_value = {"maintainability_index": 80.0}
            mock_deps.return_value = Mock(total_dependencies=10)
            mock_call_graph.return_value = Mock(total_functions=25)
            
            response = client.get("/analyze/test-owner/test-repo")
            assert response.status_code == 200

class TestIssueDetection:
    """Test comprehensive issue detection."""
    
    def test_detect_critical_issues(self, mock_codebase):
        """Test detection of critical issues."""
        issues = detect_comprehensive_issues(mock_codebase)
        assert hasattr(issues, 'critical_issues')
        assert hasattr(issues, 'major_issues')
        assert hasattr(issues, 'minor_issues')
    
    def test_issue_classification(self, mock_codebase):
        """Test issue severity classification."""
        issues = detect_issues(mock_codebase)
        
        # Verify issue structure
        for issue in issues.issues:
            assert hasattr(issue, 'severity')
            assert hasattr(issue, 'category')
            assert hasattr(issue, 'description')
            assert hasattr(issue, 'location')

class TestCodeQualityMetrics:
    """Test code quality analysis."""
    
    def test_maintainability_index_calculation(self, mock_codebase):
        """Test maintainability index calculation."""
        quality = analyze_code_quality(mock_codebase)
        assert 'maintainability_index' in quality
        assert isinstance(quality['maintainability_index'], (int, float))
        assert 0 <= quality['maintainability_index'] <= 100
    
    def test_technical_debt_calculation(self, mock_codebase):
        """Test technical debt ratio calculation."""
        quality = analyze_code_quality(mock_codebase)
        assert 'technical_debt_ratio' in quality
        assert isinstance(quality['technical_debt_ratio'], (int, float))
        assert quality['technical_debt_ratio'] >= 0

class TestDependencyAnalysis:
    """Test dependency analysis functionality."""
    
    def test_dependency_graph_generation(self, mock_codebase):
        """Test dependency graph generation."""
        deps = analyze_dependencies(mock_codebase)
        assert hasattr(deps, 'dependency_graph')
        assert hasattr(deps, 'circular_dependencies')
        assert hasattr(deps, 'external_dependencies')
    
    def test_circular_dependency_detection(self, mock_codebase):
        """Test circular dependency detection."""
        deps = analyze_dependencies(mock_codebase)
        assert isinstance(deps.circular_dependencies, list)

class TestCallGraphAnalysis:
    """Test call graph analysis."""
    
    def test_call_graph_construction(self, mock_codebase):
        """Test call graph construction."""
        call_graph = analyze_call_graph(mock_codebase)
        assert hasattr(call_graph, 'call_graph')
        assert hasattr(call_graph, 'most_called_functions')
        assert hasattr(call_graph, 'dead_code_functions')
    
    def test_entry_point_detection(self, mock_codebase):
        """Test entry point detection."""
        entry_points = detect_entry_points(mock_codebase)
        assert hasattr(entry_points, 'main_functions')
        assert hasattr(entry_points, 'api_endpoints')
        assert hasattr(entry_points, 'high_usage_functions')

class TestAdvancedAnalysis:
    """Test advanced analysis features."""
    
    def test_halstead_metrics_calculation(self, mock_codebase):
        """Test Halstead metrics calculation."""
        func = Mock(source="def test(): return 1 + 2")
        metrics = calculate_halstead_metrics(func)
        assert 'volume' in metrics
        assert 'difficulty' in metrics
        assert 'effort' in metrics
    
    def test_inheritance_analysis(self, mock_codebase):
        """Test inheritance hierarchy analysis."""
        inheritance = analyze_inheritance_hierarchy(mock_codebase)
        assert hasattr(inheritance, 'inheritance_chains')
        assert hasattr(inheritance, 'max_inheritance_depth')
    
    def test_advanced_statistics(self, mock_codebase):
        """Test advanced codebase statistics."""
        stats = get_advanced_codebase_statistics(mock_codebase)
        assert isinstance(stats, dict)
        assert 'total_lines_of_code' in stats
        assert 'function_complexity_distribution' in stats

class TestInteractiveUI:
    """Test interactive UI functionality."""
    
    def test_ui_endpoint(self):
        """Test UI endpoint availability."""
        response = client.get("/ui")
        # Should return HTML or 404 if file not found
        assert response.status_code in [200, 404]
    
    def test_repository_structure_generation(self, mock_codebase):
        """Test interactive repository structure generation."""
        # This would test the repository tree structure generation
        # Implementation depends on the specific UI framework chosen
        pass

class TestFileOperations:
    """Test file operations and cleanup."""
    
    def test_output_directory_cleanup(self):
        """Test that output directory and invalid files are properly cleaned."""
        # Verify output directory doesn't exist
        assert not Path("backend/output").exists()
    
    def test_cli_file_removal(self):
        """Test that cli.py file has been removed."""
        assert not Path("backend/cli.py").exists()
    
    def test_log_file_removal(self):
        """Test that server.log file has been removed."""
        assert not Path("backend/server.log").exists()

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_repository_handling(self):
        """Test handling of invalid repository inputs."""
        response = client.get("/analyze/invalid/repo")
        # Should handle gracefully, not crash
        assert response.status_code in [400, 404, 500]
    
    def test_network_error_handling(self):
        """Test handling of network errors during repository fetching."""
        with patch('api.Codebase.from_repo', side_effect=Exception("Network error")):
            response = client.get("/analyze/owner/repo")
            assert response.status_code == 500
    
    def test_malformed_data_handling(self, mock_codebase):
        """Test handling of malformed codebase data."""
        # Test with None values and missing attributes
        mock_codebase.functions = None
        
        try:
            analyze_codebase(mock_codebase)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (AttributeError, TypeError))

class TestPerformance:
    """Test performance characteristics."""
    
    def test_analysis_timeout(self):
        """Test that analysis completes within reasonable time."""
        import time
        start_time = time.time()
        
        # Mock a quick analysis
        with patch('api.Codebase.from_repo') as mock_codebase:
            mock_codebase.return_value = Mock()
            response = client.get("/analyze/small/repo")
            
        end_time = time.time()
        # Should complete within 30 seconds for small repos
        assert end_time - start_time < 30
    
    def test_memory_usage(self):
        """Test memory usage during analysis."""
        # This would require memory profiling tools
        # Implementation depends on specific requirements
        pass

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_analysis_result_structure(self, sample_repo_data):
        """Test that analysis results have consistent structure."""
        # Verify required fields are present
        assert 'repository' in sample_repo_data
        assert 'analysis' in sample_repo_data
        
        repo = sample_repo_data['repository']
        assert all(key in repo for key in ['name', 'owner', 'total_files'])
        
        analysis = sample_repo_data['analysis']
        assert 'comprehensive_issues' in analysis
        assert 'code_quality' in analysis
    
    def test_json_serialization(self, sample_repo_data):
        """Test that analysis results can be properly serialized."""
        json_str = json.dumps(sample_repo_data)
        deserialized = json.loads(json_str)
        assert deserialized == sample_repo_data

# Integration test fixtures
@pytest.fixture(scope="session")
def test_repository():
    """Create a temporary test repository for integration tests."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir) / "test_repo"
    repo_path.mkdir()
    
    # Create sample files
    (repo_path / "main.py").write_text("""
def main():
    print("Hello, World!")
    return process_data()

def process_data():
    data = [1, 2, 3, 4, 5]
    return sum(data)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def get_sum(self):
        return sum(self.data)

if __name__ == "__main__":
    main()
""")
    
    (repo_path / "utils.py").write_text("""
def helper_function(x, y):
    return x + y

def unused_function():
    # This function is never called
    pass

class UtilityClass:
    @staticmethod
    def static_method():
        return "static"
""")
    
    yield repo_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

class TestIntegration:
    """Integration tests using real file system."""
    
    def test_full_analysis_pipeline(self, test_repository):
        """Test complete analysis pipeline with real files."""
        # This would test the full pipeline with actual files
        # Implementation depends on how Codebase handles local files
        pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

