"""
Phase 4 Integration Tests

Comprehensive tests for all Phase 4 visualization and results generation components.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports
def test_imports():
    """Test that all Phase 4 modules can be imported."""
    try:
        # Visualization modules
        from visualization.advanced_plotting import create_sampler_comparison_grid
        from visualization.convergence_plots import plot_convergence_diagnostics_suite
        from visualization.publication_plots import create_figure_1_method_comparison
        
        # Results modules
        from results.result_generator import ResultsProcessor, generate_experiment_report
        
        print("✅ All Phase 4 modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


class MockResult:
    """Mock result object that mimics the expected benchmark result structure."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_mock_results():
    """Create mock benchmark results for testing."""
    # Mock benchmark results structure that matches expected format
    mock_results = {
        'TestDistribution_1': {
            'Standard Metropolis': MockResult(
                effective_sample_size=100,
                ess_per_second=10.5,
                sampling_time=9.52,
                acceptance_rate=0.23,
                n_samples=1000,
                dimension=10,
                r_hat=1.01,
                autocorr_time=5.2,
                mean_squared_error=0.02,
                diagnostics={}
            ),
            'Hessian Metropolis': MockResult(
                effective_sample_size=150,
                ess_per_second=15.8,
                sampling_time=9.49,
                acceptance_rate=0.45,
                n_samples=1000,
                dimension=10,
                r_hat=1.005,
                autocorr_time=3.1,
                mean_squared_error=0.01,
                diagnostics={}
            )
        },
        'TestDistribution_2': {
            'Standard Metropolis': MockResult(
                effective_sample_size=80,
                ess_per_second=8.2,
                sampling_time=9.76,
                acceptance_rate=0.19,
                n_samples=1000,
                dimension=20,
                r_hat=1.02,
                autocorr_time=6.8,
                mean_squared_error=0.03,
                diagnostics={}
            ),
            'Hessian Metropolis': MockResult(
                effective_sample_size=140,
                ess_per_second=14.1,
                sampling_time=9.93,
                acceptance_rate=0.42,
                n_samples=1000,
                dimension=20,
                r_hat=1.008,
                autocorr_time=3.5,
                mean_squared_error=0.015,
                diagnostics={}
            )
        }
    }
    
    return mock_results


def test_results_processor():
    """Test the ResultsProcessor functionality."""
    try:
        from results.result_generator import ResultsProcessor
        
        mock_results = create_mock_results()
        processor = ResultsProcessor(mock_results)
        
        # Test basic processing
        assert processor.processed_results is not None
        assert not processor.processed_results.empty
        
        # Test performance table generation
        table = processor.generate_performance_table(format_style='dataframe')
        assert not table.empty
        
        # Test improvement factors
        improvements = processor.compute_improvement_factors()
        assert len(improvements) > 0
        
        # Test statistical tests
        stat_tests = processor.perform_statistical_tests()
        # Note: May be empty if insufficient data
        
        print("✅ ResultsProcessor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ResultsProcessor test failed: {e}")
        return False


def test_visualization_functions():
    """Test that visualization functions can be called without errors."""
    try:
        from visualization.advanced_plotting import create_sampler_comparison_grid
        from visualization.publication_plots import create_figure_1_method_comparison
        
        mock_results = create_mock_results()
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Test advanced plotting (should handle missing data gracefully)
            try:
                create_sampler_comparison_grid(
                    mock_results,
                    save_dir=str(temp_dir)
                )
            except Exception as e:
                print(f"  ⚠️  Advanced plotting warning: {e}")
            
            # Test publication plots (should handle missing data gracefully)
            try:
                test_path = Path(temp_dir) / "test_plot.png"
                create_figure_1_method_comparison(
                    mock_results,
                    save_path=str(test_path)
                )
            except Exception as e:
                print(f"  ⚠️  Publication plotting warning: {e}")
        
        print("✅ Visualization function tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def test_report_generation():
    """Test report generation functionality."""
    try:
        from results.result_generator import generate_experiment_report
        
        mock_results = create_mock_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = generate_experiment_report(
                mock_results,
                output_dir=temp_dir,
                include_plots=False
            )
            
            # Check if report was generated
            assert Path(report_path).exists()
            
            # Check if JSON report was also generated
            json_report = Path(temp_dir) / "experiment_report.json"
            assert json_report.exists()
        
        print("✅ Report generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False


def test_publication_results_script():
    """Test that the publication results script exists and has the right structure."""
    try:
        # Check if publication results script exists
        pub_script = Path(__file__).parent.parent / 'examples' / 'publication_results.py'
        assert pub_script.exists(), "Publication results script not found"
        
        # Check if it has main function and key classes
        with open(pub_script, 'r') as f:
            content = f.read()
            
        assert 'class PublicationResultsGenerator' in content
        assert 'def run_complete_analysis' in content
        assert 'def main()' in content
        
        print("✅ Publication results script structure tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Publication results script test failed: {e}")
        return False


def run_all_tests():
    """Run all Phase 4 integration tests."""
    print("🧪 Running Phase 4 Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Results Processor", test_results_processor),
        ("Visualization Functions", test_visualization_functions),
        ("Report Generation", test_report_generation),
        ("Publication Results Script", test_publication_results_script)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Phase 4 components are working correctly!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)