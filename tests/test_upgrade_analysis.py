"""
Tests for the upgrade analysis functionality.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import the backend module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the graph_sitter module
sys.modules['graph_sitter'] = MagicMock()
sys.modules['graph_sitter.core'] = MagicMock()
sys.modules['graph_sitter.core.class_definition'] = MagicMock()
sys.modules['graph_sitter.core.codebase'] = MagicMock()
sys.modules['graph_sitter.core.external_module'] = MagicMock()
sys.modules['graph_sitter.core.file'] = MagicMock()
sys.modules['graph_sitter.core.function'] = MagicMock()
sys.modules['graph_sitter.core.import_resolution'] = MagicMock()
sys.modules['graph_sitter.core.symbol'] = MagicMock()
sys.modules['graph_sitter.core.context'] = MagicMock()
sys.modules['graph_sitter.enums'] = MagicMock()

from backend.api import app, UpgradeAnalysisRequest
from fastapi.testclient import TestClient

client = TestClient(app)

class TestUpgradeAnalysis(unittest.TestCase):
    """Test cases for the upgrade analysis endpoint."""
    
    @patch('backend.api.git.Repo.clone_from')
    def test_upgrade_analysis_endpoint(self, mock_clone):
        """Test that the upgrade analysis endpoint returns the expected response."""
        # Mock the git clone operation
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        
        # Test data
        test_data = {
            "repo_url": "test/repo",
            "branch": "main",
            "depth": 2,
            "options": {
                "run_tests": True,
                "create_pr": False,
                "include_dev_dependencies": True,
                "skip_major_versions": False
            }
        }
        
        # Make the request
        response = client.post("/upgrade_analysis", json=test_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        
        # Parse the response
        data = response.json()
        
        # Check the structure of the response
        self.assertEqual(data["status"], "success")
        self.assertIn("summary", data)
        self.assertIn("details", data)
        self.assertIn("upgradedDependencies", data["details"])
        self.assertIn("codeChanges", data["details"])
        self.assertIn("testResults", data["details"])
        
        # Check that the mock was called correctly
        mock_clone.assert_called_once()
        
    @patch('backend.api.git.Repo.clone_from')
    def test_upgrade_analysis_with_minimal_params(self, mock_clone):
        """Test that the upgrade analysis endpoint works with minimal parameters."""
        # Mock the git clone operation
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        
        # Test data with minimal parameters
        test_data = {
            "repo_url": "test/repo"
        }
        
        # Make the request
        response = client.post("/upgrade_analysis", json=test_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        
        # Parse the response
        data = response.json()
        
        # Check the structure of the response
        self.assertEqual(data["status"], "success")
        
    @patch('backend.api.git.Repo.clone_from')
    def test_upgrade_analysis_with_invalid_depth(self, mock_clone):
        """Test that the upgrade analysis endpoint validates the depth parameter."""
        # Test data with invalid depth
        test_data = {
            "repo_url": "test/repo",
            "depth": 5  # Invalid depth (should be 1-3)
        }
        
        # Make the request
        response = client.post("/upgrade_analysis", json=test_data)
        
        # Check the response
        self.assertEqual(response.status_code, 422)  # Validation error
        
    @patch('backend.api.git.Repo.clone_from')
    def test_upgrade_analysis_with_git_error(self, mock_clone):
        """Test that the upgrade analysis endpoint handles git errors correctly."""
        # Mock the git clone operation to raise an error
        mock_clone.side_effect = Exception("Git error")
        
        # Test data
        test_data = {
            "repo_url": "test/repo"
        }
        
        # Make the request
        response = client.post("/upgrade_analysis", json=test_data)
        
        # Check the response
        self.assertEqual(response.status_code, 500)
        
        # Parse the response
        data = response.json()
        
        # Check the error message
        self.assertIn("detail", data)
        self.assertIn("Error analyzing upgrade paths", data["detail"])

if __name__ == '__main__':
    unittest.main()
